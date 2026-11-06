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

//! Layer lifecycle, uploads, size limits, content round trips and operation
//! ids: the port of `test_layers.py`.
//!
//! Every case runs on the session the runner creates for it, and writes the
//! layer `layer` unless it names another.

use std::collections::BTreeMap;
use std::fmt;

use tonic::Code;

use crate::check::CaseResult;
use crate::client::{CallError, CreateOptions, UpdateOptions};
use crate::fixtures::{fixture, Fixture};
use crate::proto;
use crate::protocol::{
    apply_snapshot, carries_detail, decode, delta_deleting_section, delta_of_children_list,
    delta_of_entries, error_info, pack_ompf, snapshot, ErrorInfo, Scene, ERROR_DOMAIN,
    ERROR_INFO_TYPE, RETRY_INFO_TYPE,
};
use crate::runner::{Case, CaseFuture, Ctx, Profile};
use crate::{check, check_eq, expect_code};

/// The case module's name in case ids.
const FILE: &str = "layers";

// The size limits a delta is refused for: entries (struct commands, set fields
// and time samples) summed, and the ids one ReorderChildren list names.
const MAX_DELTA_ENTRIES: u32 = 99_999;
const MAX_CHILDREN_LIST_LEN: u32 = 100_000;

/// A client operation id: the recommended UUIDv4 text form.
const OPERATION_ID: &str = "3f2b6c1e-8d4a-4b7e-9c05-1a2b3c4d5e6f";

/// How many characters of a value's `Debug` form a failure message carries: a
/// folded `large` scene holds a 2 MiB value.
const REPORT_VALUE_CHARS: usize = 2048;

/// The upload RPC a parametrized case exercises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operation {
    CreateLayer,
    UpdateLayer,
}

impl Operation {
    /// Both, in the Python parametrization's order.
    const ALL: [Operation; 2] = [Operation::CreateLayer, Operation::UpdateLayer];

    /// The RPC's name: the parametrization's id.
    fn name(self) -> &'static str {
        match self {
            Operation::CreateLayer => "CreateLayer",
            Operation::UpdateLayer => "UpdateLayer",
        }
    }
}

/// The malformed payloads, with their parameter ids: not an OMPF container at
/// all; the magic and a part count of 1 with the rest of the header missing;
/// the magic and 40 bytes of `0xff`.
fn malformed_payloads() -> Vec<(&'static str, Vec<u8>)> {
    let mut all_ff = b"OMPF".to_vec();
    all_ff.extend([0xff; 40]);
    vec![
        ("not-ompf", b"not-ompf".to_vec()),
        ("OMPF-count-1-truncated", b"OMPF\x01\x00\x00\x00".to_vec()),
        ("OMPF-ff-x40", all_ff),
    ]
}

/// The malformed operation ids, with their parameter ids: over 64 characters,
/// and two with a character outside the allowed set.
fn malformed_operation_ids() -> Vec<(&'static str, String)> {
    vec![
        ("a-x65", "a".repeat(65)),
        ("op:1", "op:1".to_string()),
        ("op 1", "op 1".to_string()),
    ]
}

/// This module's cases, in definition order.
pub fn cases() -> Vec<Case> {
    let mut cases = vec![Case::new(
        FILE,
        "test_layer_lifecycle_and_generation_precondition",
        Profile::Core,
        true,
        test_layer_lifecycle_and_generation_precondition,
    )];
    for (param_id, payload) in malformed_payloads() {
        cases.push(Case::with_param(
            FILE,
            "test_malformed_content_is_rejected_without_creating_layer",
            param_id,
            Profile::Core,
            true,
            payload,
            test_malformed_content_is_rejected_without_creating_layer,
        ));
    }
    cases.push(Case::new(
        FILE,
        "test_delta_over_the_entry_limit_is_refused_delta_too_large_by_both_rpcs",
        Profile::Core,
        true,
        test_delta_over_the_entry_limit_is_refused_delta_too_large_by_both_rpcs,
    ));
    cases.push(Case::new(
        FILE,
        "test_delta_one_entry_over_the_limit_is_refused_and_one_at_it_never_delta_too_large",
        Profile::Core,
        true,
        test_delta_one_entry_over_the_limit_is_refused_and_one_at_it_never_delta_too_large,
    ));
    for children in [MAX_CHILDREN_LIST_LEN + 1, 1_000_001] {
        cases.push(Case::with_param(
            FILE,
            "test_children_list_over_the_limit_is_refused_children_list_too_long_by_both_rpcs",
            &children.to_string(),
            Profile::Core,
            true,
            children,
            test_children_list_over_the_limit_is_refused_children_list_too_long_by_both_rpcs,
        ));
    }
    for operation in Operation::ALL {
        cases.push(Case::with_param(
            FILE,
            "test_chunk_before_header_is_invalid",
            operation.name(),
            Profile::Core,
            true,
            operation,
            test_chunk_before_header_is_invalid,
        ));
    }
    for name in [
        Fixture::Initial,
        Fixture::TimeSample,
        Fixture::External,
        Fixture::Large,
    ] {
        cases.push(Case::with_param(
            FILE,
            "test_content_round_trip",
            name.name(),
            Profile::Core,
            true,
            name,
            test_content_round_trip,
        ));
    }
    cases.push(Case::new(
        FILE,
        "test_header_only_create_holds_exactly_the_layers_root_section",
        Profile::Core,
        true,
        test_header_only_create_holds_exactly_the_layers_root_section,
    ));
    cases.push(Case::new(
        FILE,
        "test_update_removing_the_layers_last_top_level_section_is_invalid_argument",
        Profile::Core,
        true,
        test_update_removing_the_layers_last_top_level_section_is_invalid_argument,
    ));
    cases.push(Case::new(
        FILE,
        "test_missing_layer_subscription_returns_not_found",
        Profile::Core,
        true,
        test_missing_layer_subscription_returns_not_found,
    ));
    cases.push(Case::new(
        FILE,
        "test_requests_without_operation_id_are_answered_unspecified",
        Profile::Core,
        true,
        test_requests_without_operation_id_are_answered_unspecified,
    ));
    cases.push(Case::new(
        FILE,
        "test_update_resend_with_operation_id_is_answered_already_committed",
        Profile::Core,
        true,
        test_update_resend_with_operation_id_is_answered_already_committed,
    ));
    cases.push(Case::new(
        FILE,
        "test_create_resend_with_operation_id_is_answered_already_committed",
        Profile::Core,
        true,
        test_create_resend_with_operation_id_is_answered_already_committed,
    ));
    cases.push(Case::new(
        FILE,
        "test_operation_id_reused_for_a_different_request_is_already_exists",
        Profile::Core,
        true,
        test_operation_id_reused_for_a_different_request_is_already_exists,
    ));
    cases.push(Case::new(
        FILE,
        "test_update_operation_id_without_client_id_is_invalid_argument",
        Profile::Core,
        true,
        test_update_operation_id_without_client_id_is_invalid_argument,
    ));
    // Stacked parametrizations: pytest runs the decorator nearest the
    // function (`operation`) as the outer loop and puts its id first.
    for operation in Operation::ALL {
        for (operation_id_param, operation_id) in malformed_operation_ids() {
            cases.push(Case::with_param(
                FILE,
                "test_malformed_operation_id_is_invalid_argument",
                &format!("{}-{operation_id_param}", operation.name()),
                Profile::Core,
                true,
                (operation, operation_id),
                test_malformed_operation_id_is_invalid_argument,
            ));
        }
    }
    cases
}

// --- Helpers.

/// A `StatLayer` request for `layer` in `session`.
fn stat_request(session: &str, layer: &str) -> proto::StatLayerRequest {
    proto::StatLayerRequest {
        session_id: session.to_string(),
        layer_name: layer.to_string(),
    }
}

/// `value`'s `Debug` form, cut to [`REPORT_VALUE_CHARS`] characters.
fn bounded_debug(value: &impl fmt::Debug) -> String {
    let text = format!("{value:?}");
    match text.char_indices().nth(REPORT_VALUE_CHARS) {
        Some((at, _)) => format!(
            "{}… ({} more characters)",
            &text[..at],
            text[at..].chars().count()
        ),
        None => text,
    }
}

/// `INVALID_ARGUMENT` carrying exactly one `ErrorInfo` `reason` in the
/// service's domain with `metadata`, each value an unsigned decimal, and no
/// `RetryInfo`: the refusal applied nothing and no re-send of the same delta
/// can succeed.
fn assert_size_refusal(
    status: &tonic::Status,
    reason: &str,
    metadata: BTreeMap<String, String>,
) -> CaseResult {
    check_eq!(status.code(), Code::InvalidArgument);
    let all_decimal = metadata
        .values()
        .all(|value| !value.is_empty() && value.chars().all(|c| c.is_ascii_digit()));
    check_eq!(
        error_info(status),
        Some(ErrorInfo {
            reason: reason.to_string(),
            domain: ERROR_DOMAIN.to_string(),
            metadata,
        })
    );
    check!(all_decimal);
    check!(
        !carries_detail(status, RETRY_INFO_TYPE)?,
        "A size refusal carries no RetryInfo"
    );
    Ok(())
}

fn assert_delta_too_large(
    status: &tonic::Status,
    struct_commands: u32,
    set_fields: u32,
    time_samples: u32,
) -> CaseResult {
    let metadata = BTreeMap::from([
        ("struct_commands".to_string(), struct_commands.to_string()),
        ("set_fields".to_string(), set_fields.to_string()),
        ("time_samples".to_string(), time_samples.to_string()),
        ("max_entries".to_string(), MAX_DELTA_ENTRIES.to_string()),
    ]);
    assert_size_refusal(status, "DELTA_TOO_LARGE", metadata)?;
    let message = status.message();
    let total = u64::from(struct_commands) + u64::from(set_fields) + u64::from(time_samples);
    check!(
        message.contains(&total.to_string()) && message.contains(&MAX_DELTA_ENTRIES.to_string()),
        "{message}"
    );
    check!(message.contains("split"), "The message names the remedy");
    Ok(())
}

fn assert_children_list_too_long(status: &tonic::Status, actual: u32) -> CaseResult {
    let metadata = BTreeMap::from([
        ("limit".to_string(), MAX_CHILDREN_LIST_LEN.to_string()),
        ("actual".to_string(), actual.to_string()),
    ]);
    assert_size_refusal(status, "CHILDREN_LIST_TOO_LONG", metadata)
}

// --- Cases.

fn test_layer_lifecycle_and_generation_precondition(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let first = client.create(session, fixture(Fixture::Initial)).await?;
        check!(first.object_id > 0);
        let response = client
            .update(session, first.object_id, fixture(Fixture::Update))
            .await?;
        check_eq!(response.sequence_num, 2);
        check!(response.server_timestamp_microseconds_since_epoch > 0);
        let metadata = client.stat_layer(stat_request(session, "layer")).await?;
        check!(metadata.created_at > 0 && metadata.last_modified_at >= metadata.created_at);
        let second = client.create(session, fixture(Fixture::Initial)).await?;
        check!(second.object_id > first.object_id);
        expect_code!(
            client
                .update(session, first.object_id, fixture(Fixture::Update))
                .await,
            Code::FailedPrecondition
        );
        check_eq!(
            client
                .update(session, second.object_id, b"")
                .await?
                .sequence_num,
            2
        );
        client
            .delete_layer(proto::DeleteLayerRequest {
                session_id: session.to_string(),
                layer_name: "layer".to_string(),
            })
            .await?;
        expect_code!(
            client.stat_layer(stat_request(session, "layer")).await,
            Code::NotFound
        );
        Ok(())
    })
}

fn test_malformed_content_is_rejected_without_creating_layer(
    ctx: &mut Ctx,
    payload: Vec<u8>,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        expect_code!(
            client.create(session, &payload).await,
            Code::InvalidArgument
        );
        expect_code!(
            client.stat_layer(stat_request(session, "layer")).await,
            Code::NotFound
        );
        Ok(())
    })
}

fn test_delta_over_the_entry_limit_is_refused_delta_too_large_by_both_rpcs(
    ctx: &mut Ctx,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        // 60,000 struct commands + 60,000 set fields: each under 100,000, the
        // sum over 99,999. Built here, not a committed fixture.
        let payload = pack_ompf(&delta_of_entries(60_000, 60_000));
        let oversized = CreateOptions {
            layer: "oversized".to_string(),
            ..CreateOptions::default()
        };
        let status = expect_code!(
            client.create_with(session, &payload, &oversized).await,
            Code::InvalidArgument
        );
        assert_delta_too_large(&status, 60_000, 60_000, 0)?;
        expect_code!(
            client.stat_layer(stat_request(session, "oversized")).await,
            Code::NotFound,
            "No layer was created"
        );

        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let status = expect_code!(
            client.update(session, created.object_id, &payload).await,
            Code::InvalidArgument
        );
        assert_delta_too_large(&status, 60_000, 60_000, 0)?;
        // Nothing applied: the next commit is the layer's second.
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

fn test_delta_one_entry_over_the_limit_is_refused_and_one_at_it_never_delta_too_large(
    ctx: &mut Ctx,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let status = expect_code!(
            client
                .update(
                    session,
                    created.object_id,
                    &pack_ompf(&delta_of_entries(0, 100_000))
                )
                .await,
            Code::InvalidArgument
        );
        assert_delta_too_large(&status, 0, 100_000, 0)?;
        // Exactly the limit is never refused DELTA_TOO_LARGE. These entries'
        // content may be accepted or refused on its own merits; this checks
        // only that a refusal, if any, does not carry that reason. A local
        // failure is no answer from the server, so it fails the case.
        match client
            .update(
                session,
                created.object_id,
                &pack_ompf(&delta_of_entries(0, 99_999)),
            )
            .await
        {
            Ok(_) => {}
            Err(CallError::Status(rejected)) => {
                // An ErrorInfo-typed detail must decode, so a garbled one
                // never reads as "no ErrorInfo".
                if carries_detail(&rejected, ERROR_INFO_TYPE)? {
                    check!(
                        error_info(&rejected).is_some_and(|info| info.reason != "DELTA_TOO_LARGE")
                    );
                }
            }
            Err(local @ CallError::Local(_)) => return Err(local.into()),
        }
        Ok(())
    })
}

fn test_children_list_over_the_limit_is_refused_children_list_too_long_by_both_rpcs(
    ctx: &mut Ctx,
    children: u32,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let payload = pack_ompf(&delta_of_children_list(children));
        let oversized = CreateOptions {
            layer: "oversized".to_string(),
            ..CreateOptions::default()
        };
        let status = expect_code!(
            client.create_with(session, &payload, &oversized).await,
            Code::InvalidArgument
        );
        assert_children_list_too_long(&status, children)?;
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let status = expect_code!(
            client.update(session, created.object_id, &payload).await,
            Code::InvalidArgument
        );
        assert_children_list_too_long(&status, children)?;
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

fn test_chunk_before_header_is_invalid(ctx: &mut Ctx, operation: Operation) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let data = fixture(Fixture::Initial).to_vec();
        let result = match operation {
            Operation::CreateLayer => client
                .create_layer_raw(vec![proto::CreateLayerRequest {
                    message_type: Some(proto::create_layer_request::MessageType::Chunk(
                        proto::CreateLayerChunk { data },
                    )),
                }])
                .await
                .map(drop),
            Operation::UpdateLayer => client
                .update_layer_raw(vec![proto::UpdateLayerRequest {
                    message_type: Some(proto::update_layer_request::MessageType::Chunk(
                        proto::UpdateLayerChunk { data },
                    )),
                }])
                .await
                .map(drop),
        };
        expect_code!(result, Code::InvalidArgument);
        Ok(())
    })
}

fn test_content_round_trip(ctx: &mut Ctx, name: Fixture) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let payload = fixture(name);
        let options = CreateOptions {
            chunk_size: 4096,
            ..CreateOptions::default()
        };
        let created = client.create_with(session, payload, &options).await?;
        let mut stream = client.subscribe(session, 0, 0).await?;
        let (header, returned) = stream.delivery().await?;
        check_eq!(
            (header.object_id, header.sequence_num),
            (created.object_id, 1)
        );
        let sent_signature = decode(payload)?;
        let received_signature = snapshot(&returned, 0)?;
        check_eq!(
            received_signature.source_format,
            sent_signature.source_format
        );
        let expected = apply_snapshot(&sent_signature, None)?;
        let actual = apply_snapshot(&received_signature, None)?;
        check!(
            actual == expected,
            "the folded catch-up differs from the folded upload\n  actual: {}\nexpected: {}",
            bounded_debug(&actual),
            bounded_debug(&expected)
        );
        stream.caught_up(created.object_id, 1).await?;
        Ok(())
    })
}

fn test_header_only_create_holds_exactly_the_layers_root_section(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        // A header with no content creates the layer holding only its root
        // section: section 1 under 0, empty name, section type 7.
        let created = client.create(session, b"").await?;
        let mut stream = client.subscribe(session, 0, 0).await?;
        let (header, returned) = stream.delivery().await?;
        check_eq!(
            (header.object_id, header.sequence_num),
            (created.object_id, 1)
        );
        let root_only = Scene {
            sections: BTreeMap::from([(1, (0, String::new(), 7, 0))]),
            fields: BTreeMap::new(),
            samples: BTreeMap::new(),
        };
        check_eq!(apply_snapshot(&snapshot(&returned, 0)?, None)?, root_only);
        stream.caught_up(created.object_id, 1).await?;
        Ok(())
    })
}

fn test_update_removing_the_layers_last_top_level_section_is_invalid_argument(
    ctx: &mut Ctx,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        // The `initial` fixture's only section under 0 is section 1.
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        expect_code!(
            client
                .update(
                    session,
                    created.object_id,
                    &pack_ompf(&delta_deleting_section(0, 1))
                )
                .await,
            Code::InvalidArgument
        );
        // Nothing applied: the next commit is the layer's second.
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

fn test_missing_layer_subscription_returns_not_found(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let mut stream = ctx.client.subscribe(&ctx.session, 0, 0).await?;
        expect_code!(stream.next().await, Code::NotFound);
        Ok(())
    })
}

fn test_requests_without_operation_id_are_answered_unspecified(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        check_eq!(
            created.operation_outcome,
            proto::OperationOutcome::Unspecified as i32
        );
        let updated = client
            .update(session, created.object_id, fixture(Fixture::Update))
            .await?;
        check_eq!(
            updated.operation_outcome,
            proto::OperationOutcome::Unspecified as i32
        );
        Ok(())
    })
}

fn test_update_resend_with_operation_id_is_answered_already_committed(
    ctx: &mut Ctx,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let first = client
            .update_with(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &UpdateOptions {
                    timestamp: 111,
                    operation_id: OPERATION_ID.to_string(),
                    ..UpdateOptions::default()
                },
            )
            .await?;
        check_eq!(
            first.operation_outcome,
            proto::OperationOutcome::Applied as i32
        );
        // A re-send whose response was lost: same id and bytes, fresh
        // timestamp.
        let again = client
            .update_with(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &UpdateOptions {
                    timestamp: 222,
                    operation_id: OPERATION_ID.to_string(),
                    ..UpdateOptions::default()
                },
            )
            .await?;
        check_eq!(
            again.operation_outcome,
            proto::OperationOutcome::AlreadyCommitted as i32
        );
        check_eq!(again.sequence_num, first.sequence_num);
        check_eq!(
            again.server_timestamp_microseconds_since_epoch,
            first.server_timestamp_microseconds_since_epoch
        );
        // The re-send applied nothing: the next commit is the very next
        // sequence.
        check_eq!(
            client
                .update(session, created.object_id, b"")
                .await?
                .sequence_num,
            first.sequence_num + 1
        );
        Ok(())
    })
}

fn test_create_resend_with_operation_id_is_answered_already_committed(
    ctx: &mut Ctx,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let with_id = CreateOptions {
            operation_id: OPERATION_ID.to_string(),
            ..CreateOptions::default()
        };
        let first = client
            .create_with(session, fixture(Fixture::Initial), &with_id)
            .await?;
        check_eq!(
            first.operation_outcome,
            proto::OperationOutcome::Applied as i32
        );
        let again = client
            .create_with(session, fixture(Fixture::Initial), &with_id)
            .await?;
        check_eq!(
            again.operation_outcome,
            proto::OperationOutcome::AlreadyCommitted as i32
        );
        check_eq!(again.object_id, first.object_id);
        // The layer was not replaced by its own re-send: the original object
        // takes writes.
        check_eq!(
            client
                .update(session, first.object_id, b"")
                .await?
                .sequence_num,
            2
        );
        Ok(())
    })
}

fn test_operation_id_reused_for_a_different_request_is_already_exists(
    ctx: &mut Ctx,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let create_with_id = CreateOptions {
            operation_id: OPERATION_ID.to_string(),
            ..CreateOptions::default()
        };
        let update_with_id = UpdateOptions {
            operation_id: OPERATION_ID.to_string(),
            ..UpdateOptions::default()
        };
        let created = client
            .create_with(session, fixture(Fixture::Initial), &create_with_id)
            .await?;
        expect_code!(
            client
                .create_with(session, fixture(Fixture::TimeSample), &create_with_id)
                .await,
            Code::AlreadyExists
        );
        client
            .update_with(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &update_with_id,
            )
            .await?;
        expect_code!(
            client
                .update_with(session, created.object_id, b"", &update_with_id)
                .await,
            Code::AlreadyExists
        );
        // Neither refusal changed the layer.
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

fn test_update_operation_id_without_client_id_is_invalid_argument(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let created = client
            .create_with(
                session,
                fixture(Fixture::Initial),
                &CreateOptions {
                    operation_id: OPERATION_ID.to_string(),
                    ..CreateOptions::default()
                },
            )
            .await?;
        check_eq!(
            created.operation_outcome,
            proto::OperationOutcome::Applied as i32
        );
        // An operation id is guarded per (object, client_id): it needs a
        // client id.
        expect_code!(
            client
                .update_with(
                    session,
                    created.object_id,
                    fixture(Fixture::Update),
                    &UpdateOptions {
                        client_id: String::new(),
                        operation_id: OPERATION_ID.to_string(),
                        ..UpdateOptions::default()
                    },
                )
                .await,
            Code::InvalidArgument
        );
        // Refused before anything was applied; an empty client id without an
        // operation id is still accepted.
        let without_client_id = UpdateOptions {
            client_id: String::new(),
            ..UpdateOptions::default()
        };
        check_eq!(
            client
                .update_with(session, created.object_id, b"", &without_client_id)
                .await?
                .sequence_num,
            2
        );
        Ok(())
    })
}

fn test_malformed_operation_id_is_invalid_argument(
    ctx: &mut Ctx,
    (operation, operation_id): (Operation, String),
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = ctx.session.as_str();
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let result = match operation {
            Operation::CreateLayer => client
                .create_with(
                    session,
                    fixture(Fixture::Initial),
                    &CreateOptions {
                        operation_id,
                        ..CreateOptions::default()
                    },
                )
                .await
                .map(drop),
            Operation::UpdateLayer => client
                .update_with(
                    session,
                    created.object_id,
                    fixture(Fixture::Update),
                    &UpdateOptions {
                        operation_id,
                        ..UpdateOptions::default()
                    },
                )
                .await
                .map(drop),
        };
        expect_code!(result, Code::InvalidArgument);
        // Refused before anything was applied.
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
