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

//! Chained `UpdateLayer` writes (`UpdateLayerRequest.chained_header`): the
//! port of `test_chained.py`.
//!
//! A chained call names its predecessor, the same `client_id`'s previous
//! write to the object, and commits only directly after it, at most once.
//! Every case runs one writer's chain on a fresh layer; sequence numbers prove
//! the commit order.

use std::time::Duration;

use tonic::Code;

use crate::check::CaseResult;
use crate::client::{ChainOptions, Pending, UpdateOptions};
use crate::fixtures::{fixture, Fixture};
use crate::proto;
use crate::protocol::{
    carries_detail, delta_of_children_list, delta_of_entries, error_info, pack_ompf, ErrorInfo,
    ERROR_DOMAIN, ERROR_INFO_TYPE, RETRY_INFO_TYPE,
};
use crate::runner::{Case, CaseFuture, Ctx, Profile};
use crate::{check, check_eq, expect_code};

/// The case module's name in case ids.
const FILE: &str = "chained";

/// How long a call sent ahead of its predecessor is watched to still be
/// waiting: the observation itself, not a synchronization (an implementation
/// that commits it at once fails here).
const STILL_WAITING: Duration = Duration::from_millis(300);

/// This module's cases, in definition order.
pub fn cases() -> Vec<Case> {
    vec![
        Case::new(
            FILE,
            "test_ping_reports_operation_ids_and_chained_writes",
            Profile::Core,
            false,
            test_ping_reports_operation_ids_and_chained_writes,
        ),
        Case::new(
            FILE,
            "test_chained_writes_sent_in_order_commit_in_chain_order",
            Profile::Core,
            true,
            test_chained_writes_sent_in_order_commit_in_chain_order,
        ),
        Case::new(
            FILE,
            "test_chained_writes_sent_out_of_order_commit_in_chain_order",
            Profile::Core,
            true,
            test_chained_writes_sent_out_of_order_commit_in_chain_order,
        ),
        Case::new(
            FILE,
            "test_chained_resend_of_the_latest_write_is_answered_already_committed",
            Profile::Core,
            true,
            test_chained_resend_of_the_latest_write_is_answered_already_committed,
        ),
        Case::new(
            FILE,
            "test_waiting_chained_call_is_refused_aborted_before_its_deadline",
            Profile::Core,
            true,
            test_waiting_chained_call_is_refused_aborted_before_its_deadline,
        ),
        Case::new(
            FILE,
            "test_no_wait_call_whose_predecessor_was_passed_is_refused_at_once",
            Profile::Core,
            true,
            test_no_wait_call_whose_predecessor_was_passed_is_refused_at_once,
        ),
        Case::new(
            FILE,
            "test_cancelled_waiting_chained_call_applies_nothing",
            Profile::Core,
            true,
            test_cancelled_waiting_chained_call_applies_nothing,
        ),
        Case::new(
            FILE,
            "test_predecessor_lost_before_its_stream_ended_and_resent_releases_its_successor",
            Profile::Core,
            true,
            test_predecessor_lost_before_its_stream_ended_and_resent_releases_its_successor,
        ),
        Case::new(
            FILE,
            "test_invalid_chained_call_carries_invalid_update",
            Profile::Core,
            true,
            test_invalid_chained_call_carries_invalid_update,
        ),
        Case::new(
            FILE,
            "test_size_refused_chained_call_carries_its_size_reason_in_place_of_invalid_update",
            Profile::Core,
            true,
            test_size_refused_chained_call_carries_its_size_reason_in_place_of_invalid_update,
        ),
        Case::new(
            FILE,
            "test_chained_writes_across_replica_endpoints_commit_in_chain_order",
            Profile::MultiReplica,
            true,
            test_chained_writes_across_replica_endpoints_commit_in_chain_order,
        ),
    ]
}

// --- Helpers.

/// `N` chain operation ids, `<prefix>-1` to `<prefix>-N`: UUID-free but
/// unique to the case's fresh layer.
fn ops<const N: usize>(prefix: &str) -> [String; N] {
    std::array::from_fn(|i| format!("{prefix}-{}", i + 1))
}

fn assert_applied(response: &proto::UpdateLayerResponse, sequence: u64) -> CaseResult {
    check_eq!(
        response.operation_outcome,
        proto::OperationOutcome::Applied as i32
    );
    check_eq!(response.sequence_num, sequence);
    Ok(())
}

/// Waits [`STILL_WAITING`], then checks that none of `pending` has ended.
async fn assert_still_waiting(pending: &[&Pending<proto::UpdateLayerResponse>]) -> CaseResult {
    tokio::time::sleep(STILL_WAITING).await;
    for call in pending {
        check!(
            !call.is_finished(),
            "A call whose predecessor has not committed must wait"
        );
    }
    Ok(())
}

fn assert_predecessor_not_committed(
    status: &tonic::Status,
    metadata: &[(&str, &str)],
) -> CaseResult {
    check_eq!(status.code(), Code::Aborted);
    check_eq!(
        error_info(status),
        Some(ErrorInfo::new(
            "PREDECESSOR_NOT_COMMITTED",
            ERROR_DOMAIN,
            metadata
        )),
        "ABORTED carries the predecessor refusal's ErrorInfo"
    );
    check!(
        !carries_detail(status, RETRY_INFO_TYPE)?,
        "The refusal carries no RetryInfo"
    );
    Ok(())
}

/// The default [`ChainOptions`] with `no_wait` set.
fn no_wait() -> ChainOptions {
    ChainOptions {
        no_wait: true,
        ..ChainOptions::default()
    }
}

// --- Cases.

fn test_ping_reports_operation_ids_and_chained_writes(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let response = ctx
            .client
            .ping(proto::PingRequest {
                client_timestamp_microseconds_since_epoch: 7,
            })
            .await?;
        check_eq!(response.client_timestamp_microseconds_since_epoch, 7);
        check_eq!(response.operation_ids, true);
        check_eq!(response.chained_writes, true);
        Ok(())
    })
}

fn test_chained_writes_sent_in_order_commit_in_chain_order(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let chain: [String; 3] = ops("in-order");
        for (i, operation_id) in chain.iter().enumerate() {
            let previous = if i > 0 { chain[i - 1].as_str() } else { "" };
            let response = client
                .chained_update(
                    session,
                    created.object_id,
                    fixture(Fixture::Update),
                    operation_id,
                    previous,
                )
                .await?;
            assert_applied(&response, i as u64 + 2)?;
        }
        Ok(())
    })
}

fn test_chained_writes_sent_out_of_order_commit_in_chain_order(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let [first, second, third] = ops("out-of-order");
        let third_call = client
            .chained_update_async(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &third,
                &second,
            )
            .await;
        let second_call = client
            .chained_update_async(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &second,
                &first,
            )
            .await;
        assert_still_waiting(&[&third_call, &second_call]).await?;
        assert_applied(
            &client
                .chained_update(
                    session,
                    created.object_id,
                    fixture(Fixture::Update),
                    &first,
                    "",
                )
                .await?,
            2,
        )?;
        assert_applied(&second_call.result().await?, 3)?;
        assert_applied(&third_call.result().await?, 4)?;
        Ok(())
    })
}

fn test_chained_resend_of_the_latest_write_is_answered_already_committed(
    ctx: &mut Ctx,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let [first, second] = ops("resend");
        client
            .chained_update(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &first,
                "",
            )
            .await?;
        let committed = client
            .chained_update(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &second,
                &first,
            )
            .await?;
        let again = client
            .chained_update_with(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &second,
                &first,
                &no_wait(),
            )
            .await?;
        check_eq!(
            again.operation_outcome,
            proto::OperationOutcome::AlreadyCommitted as i32
        );
        check_eq!(again.sequence_num, committed.sequence_num);
        // The re-send applied nothing.
        check_eq!(
            client
                .update(session, created.object_id, b"")
                .await?
                .sequence_num,
            4
        );
        Ok(())
    })
}

fn test_waiting_chained_call_is_refused_aborted_before_its_deadline(
    ctx: &mut Ctx,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let [first, second, third] = ops("deadline");
        client
            .chained_update(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &first,
                "",
            )
            .await?;
        // `second` never commits: `third` waits until its margin before its
        // deadline.
        let three_seconds = ChainOptions {
            timeout: Some(Duration::from_secs(3)),
            ..ChainOptions::default()
        };
        let status = expect_code!(
            client
                .chained_update_async_with(
                    session,
                    created.object_id,
                    fixture(Fixture::Update),
                    &third,
                    &second,
                    &three_seconds,
                )
                .await
                .result()
                .await,
            Code::Aborted
        );
        assert_predecessor_not_committed(
            &status,
            &[
                ("last_record", "operation"),
                ("last_operation_id", first.as_str()),
            ],
        )?;
        check_eq!(
            client
                .update(session, created.object_id, b"")
                .await?
                .sequence_num,
            3
        );
        Ok(())
    })
}

fn test_no_wait_call_whose_predecessor_was_passed_is_refused_at_once(
    ctx: &mut Ctx,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let [first, second, third] = ops("passed");
        // Staged with ordinary operation-id commits: the record has passed
        // `first`.
        for operation_id in [&first, &second, &third] {
            client
                .update_with(
                    session,
                    created.object_id,
                    fixture(Fixture::Update),
                    &UpdateOptions {
                        operation_id: operation_id.clone(),
                        ..UpdateOptions::default()
                    },
                )
                .await?;
        }
        let status = expect_code!(
            client
                .chained_update_with(
                    session,
                    created.object_id,
                    fixture(Fixture::Update),
                    &second,
                    &first,
                    &no_wait(),
                )
                .await,
            Code::Aborted
        );
        assert_predecessor_not_committed(
            &status,
            &[
                ("last_record", "operation"),
                ("last_operation_id", third.as_str()),
            ],
        )?;
        let status = expect_code!(
            client
                .chained_update_with(
                    session,
                    created.object_id,
                    fixture(Fixture::Update),
                    "never",
                    "unknown",
                    &ChainOptions {
                        client_id: "other-writer".to_string(),
                        no_wait: true,
                        ..ChainOptions::default()
                    },
                )
                .await,
            Code::Aborted
        );
        assert_predecessor_not_committed(&status, &[("last_record", "absent")])?;
        check_eq!(
            client
                .update(session, created.object_id, b"")
                .await?
                .sequence_num,
            5
        );
        Ok(())
    })
}

fn test_cancelled_waiting_chained_call_applies_nothing(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let [first, second] = ops("cancel");
        let waiting = client
            .chained_update_async(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &second,
                &first,
            )
            .await;
        assert_still_waiting(&[&waiting]).await?;
        check!(
            waiting.cancel().await,
            "The waiting call must still be cancellable"
        );
        assert_applied(
            &client
                .chained_update(
                    session,
                    created.object_id,
                    fixture(Fixture::Update),
                    &first,
                    "",
                )
                .await?,
            2,
        )?;
        // Its own re-send applies: the cancelled call never committed.
        assert_applied(
            &client
                .chained_update_with(
                    session,
                    created.object_id,
                    fixture(Fixture::Update),
                    &second,
                    &first,
                    &no_wait(),
                )
                .await?,
            3,
        )?;
        Ok(())
    })
}

fn test_predecessor_lost_before_its_stream_ended_and_resent_releases_its_successor(
    ctx: &mut Ctx,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let [first, second] = ops("lost");
        let successor = client
            .chained_update_async(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &second,
                &first,
            )
            .await;
        let lost = client
            .unended_update_async(client.chained_requests(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &first,
                "",
                &ChainOptions::default(),
            ))
            .await;
        assert_still_waiting(&[&successor, &lost]).await?;
        check!(
            lost.cancel().await,
            "The unended predecessor must still be cancellable"
        );
        check!(
            !successor.is_finished(),
            "The successor waits for its predecessor's re-send"
        );
        assert_applied(
            &client
                .chained_update(
                    session,
                    created.object_id,
                    fixture(Fixture::Update),
                    &first,
                    "",
                )
                .await?,
            2,
        )?;
        assert_applied(&successor.result().await?, 3)?;
        Ok(())
    })
}

fn test_invalid_chained_call_carries_invalid_update(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let status = expect_code!(
            client
                .chained_update(
                    session,
                    created.object_id,
                    fixture(Fixture::Update),
                    "same",
                    "same",
                )
                .await,
            Code::InvalidArgument
        );
        check_eq!(
            error_info(&status),
            Some(ErrorInfo::new("INVALID_UPDATE", ERROR_DOMAIN, &[]))
        );
        // The unchained INVALID_ARGUMENT keeps no ErrorInfo.
        let status = expect_code!(
            client
                .update_with(
                    session,
                    created.object_id,
                    b"",
                    &UpdateOptions {
                        operation_id: "op 1".to_string(),
                        ..UpdateOptions::default()
                    },
                )
                .await,
            Code::InvalidArgument
        );
        check!(!carries_detail(&status, ERROR_INFO_TYPE)?);
        check_eq!(
            client
                .update(session, created.object_id, b"")
                .await?
                .sequence_num,
            2
        );
        Ok(())
    })
}

fn test_size_refused_chained_call_carries_its_size_reason_in_place_of_invalid_update(
    ctx: &mut Ctx,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        // 60,000 struct commands + 60,000 set fields: over the 99,999-entry
        // limit.
        let status = expect_code!(
            client
                .chained_update(
                    session,
                    created.object_id,
                    &pack_ompf(&delta_of_entries(60_000, 60_000)),
                    "big",
                    "",
                )
                .await,
            Code::InvalidArgument
        );
        check_eq!(
            error_info(&status),
            Some(ErrorInfo::new(
                "DELTA_TOO_LARGE",
                ERROR_DOMAIN,
                &[
                    ("struct_commands", "60000"),
                    ("set_fields", "60000"),
                    ("time_samples", "0"),
                    ("max_entries", "99999"),
                ],
            )),
            "A size refusal keeps its own reason on a chained call"
        );
        check!(!carries_detail(&status, RETRY_INFO_TYPE)?);
        // A ReorderChildren list one id over the 100,000-id limit.
        let status = expect_code!(
            client
                .chained_update(
                    session,
                    created.object_id,
                    &pack_ompf(&delta_of_children_list(100_001)),
                    "long",
                    "",
                )
                .await,
            Code::InvalidArgument
        );
        check_eq!(
            error_info(&status),
            Some(ErrorInfo::new(
                "CHILDREN_LIST_TOO_LONG",
                ERROR_DOMAIN,
                &[("limit", "100000"), ("actual", "100001")],
            ))
        );
        check!(!carries_detail(&status, RETRY_INFO_TYPE)?);
        check_eq!(
            client
                .update(session, created.object_id, b"")
                .await?
                .sequence_num,
            2
        );
        Ok(())
    })
}

/// The peer client is the runner's (`Ctx::peer`), which closes it after the
/// case as the Python test's `finally` did.
fn test_chained_writes_across_replica_endpoints_commit_in_chain_order(
    ctx: &mut Ctx,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let peer = ctx.peer()?;
        let session = ctx.session.as_str();
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let [first, second, third] = ops("replicas");
        let third_call = peer
            .chained_update_async(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &third,
                &second,
            )
            .await;
        let second_call = client
            .chained_update_async(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &second,
                &first,
            )
            .await;
        assert_still_waiting(&[&third_call, &second_call]).await?;
        assert_applied(
            &peer
                .chained_update(
                    session,
                    created.object_id,
                    fixture(Fixture::Update),
                    &first,
                    "",
                )
                .await?,
            2,
        )?;
        assert_applied(&second_call.result().await?, 3)?;
        assert_applied(&third_call.result().await?, 4)?;
        Ok(())
    })
}
