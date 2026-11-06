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

//! `SubscribeReadLayer` delivery and catch-up, and `JoinSession` presence and
//! keepalives: the port of `test_streams.py`.
//!
//! Every stream a case opens stays open until the case ends (dropping a
//! [`Stream`](crate::client::Stream) cancels its call), unless the case
//! closes it, as the Python suite kept each stream until the client closed.
//! Likewise each join's [`JoinSender`](crate::client::JoinSender) is held for
//! the whole case even when the case sends nothing: the Python request
//! generator never ended its stream, so no join here half-closes.

use std::collections::BTreeMap;
use std::time::Duration;

use tokio::time::Instant;
use tonic::Code;

use crate::check::{CaseError, CaseResult};
use crate::client::{Stream, SubscribeOptions, UpdateOptions};
use crate::fixtures::{capture, capture_golden, fixture, Fixture};
use crate::proto;
use crate::proto::join_session_response::ResponseType;
use crate::protocol::{apply_snapshot, decode, field_order, sample_key, snapshot, Scene};
use crate::runner::{Case, CaseFuture, Ctx, Profile};
use crate::{check, check_eq, expect_code};

const FILE: &str = "streams";

/// This module's cases, in definition order.
pub fn cases() -> Vec<Case> {
    // Every case here is core and runs in a session of its own.
    fn case(name: &'static str, run: for<'a> fn(&'a mut Ctx) -> CaseFuture<'a>) -> Case {
        Case::new(FILE, name, Profile::Core, true, run)
    }
    vec![
        case(
            "test_live_delivery_and_self_echo_suppression",
            test_live_delivery_and_self_echo_suppression,
        ),
        case(
            "test_empty_update_is_delivered_with_commit_timestamp",
            test_empty_update_is_delivered_with_commit_timestamp,
        ),
        case(
            "test_overwrite_rebootstraps_subscription",
            test_overwrite_rebootstraps_subscription,
        ),
        case(
            "test_reconnect_folds_fields_and_time_samples",
            test_reconnect_folds_fields_and_time_samples,
        ),
        case(
            "test_captured_traffic_catchup_preserves_client_state",
            test_captured_traffic_catchup_preserves_client_state,
        ),
        case(
            "test_raw_structural_edits_have_independent_expected_state",
            test_raw_structural_edits_have_independent_expected_state,
        ),
        case(
            "test_sibling_name_rules_have_independent_expected_state",
            test_sibling_name_rules_have_independent_expected_state,
        ),
        case(
            "test_presence_join_message_leave",
            test_presence_join_message_leave,
        ),
        case(
            "test_presence_keepalive_is_advertised_on_opt_in_and_never_delivered",
            test_presence_keepalive_is_advertised_on_opt_in_and_never_delivered,
        ),
        case(
            "test_presence_silent_keepalive_join_ends_unavailable_after_leaving",
            test_presence_silent_keepalive_join_ends_unavailable_after_leaving,
        ),
    ]
}

/// Lowercase hex of `bytes`: a signature's inline `valueOrExtHash`.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn test_live_delivery_and_self_echo_suppression(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = &ctx.session;
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let mut stream = client
            .subscribe_with(
                session,
                created.object_id,
                1,
                &SubscribeOptions {
                    client_id: "self".into(),
                    ..SubscribeOptions::default()
                },
            )
            .await?;
        stream.caught_up(created.object_id, 1).await?;
        let own = client
            .update_with(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &UpdateOptions {
                    client_id: "self".into(),
                    ..UpdateOptions::default()
                },
            )
            .await?;
        check_eq!(own.sequence_num, 2);
        // A second writer's commit is an ordering barrier: a broken self-echo
        // filter delivers seq 2 before seq 3 and fails, without a silence sleep.
        let response = client
            .update_with(
                session,
                created.object_id,
                fixture(Fixture::Update),
                &UpdateOptions {
                    client_id: "other".into(),
                    ..UpdateOptions::default()
                },
            )
            .await?;
        let (header, payload) = stream.delivery().await?;
        check_eq!(header.sequence_num, response.sequence_num);
        check_eq!(response.sequence_num, 3);
        check_eq!(header.object_id, created.object_id);
        check_eq!(header.client_timestamp_microseconds_since_epoch, 123456);
        check_eq!(
            header.server_timestamp_microseconds_since_epoch,
            response.server_timestamp_microseconds_since_epoch
        );
        check!(
            payload.as_slice() == fixture(Fixture::Update),
            "Steady-state delivery must preserve the writer's OMPF"
        );
        Ok(())
    })
}

fn test_empty_update_is_delivered_with_commit_timestamp(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = &ctx.session;
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let mut stream = client.subscribe(session, created.object_id, 1).await?;
        stream.caught_up(created.object_id, 1).await?;
        let response = client.update(session, created.object_id, b"").await?;
        let (header, payload) = stream.delivery().await?;
        check_eq!(header.sequence_num, response.sequence_num);
        check_eq!(response.sequence_num, 2);
        if !payload.is_empty() {
            let signature = decode(&payload)?;
            check!(
                signature.struct_commands.is_empty()
                    && signature.set_fields.is_empty()
                    && signature.time_samples.is_empty()
            );
        }
        check_eq!(
            header.server_timestamp_microseconds_since_epoch,
            response.server_timestamp_microseconds_since_epoch
        );
        Ok(())
    })
}

fn test_overwrite_rebootstraps_subscription(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = &ctx.session;
        let first = client.create(session, fixture(Fixture::Initial)).await?;
        let mut stream = client.subscribe(session, first.object_id, 1).await?;
        stream.caught_up(first.object_id, 1).await?;
        let second = client.create(session, fixture(Fixture::TimeSample)).await?;
        let (header, payload) = stream.delivery().await?;
        check_eq!(header.object_id, second.object_id);
        check!(second.object_id != first.object_id);
        check_eq!(header.sequence_num, 1);
        check_eq!(
            apply_snapshot(&snapshot(&payload, 0)?, None)?,
            apply_snapshot(&decode(fixture(Fixture::TimeSample))?, None)?
        );
        stream.caught_up(second.object_id, 1).await?;
        client
            .delete_layer(proto::DeleteLayerRequest {
                session_id: session.clone(),
                layer_name: "layer".into(),
            })
            .await?;
        expect_code!(stream.next().await, Code::NotFound);
        Ok(())
    })
}

fn test_reconnect_folds_fields_and_time_samples(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = &ctx.session;
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let mut baseline_stream = client.subscribe(session, 0, 0).await?;
        let (_, baseline_payload) = baseline_stream.delivery().await?;
        baseline_stream.caught_up(created.object_id, 1).await?;
        baseline_stream.close();
        client
            .update(session, created.object_id, fixture(Fixture::Update))
            .await?;
        let response = client
            .update(session, created.object_id, fixture(Fixture::UpdateTime))
            .await?;
        let mut stream = client.subscribe(session, created.object_id, 1).await?;
        let (header, payload) = stream.delivery().await?;
        check_eq!(header.sequence_num, response.sequence_num);
        check_eq!(response.sequence_num, 3);
        let signature = snapshot(&payload, 1)?;
        check!(signature.is_diff && signature.base_version == 1);
        let baseline = apply_snapshot(&snapshot(&baseline_payload, 0)?, None)?;
        let actual = apply_snapshot(&signature, Some(&baseline))?;
        // Fixed expected values, independent of a second server read.
        check_eq!(
            actual.fields.get(&(5, "value".to_string())),
            Some(&(hex(b"updated"), 0))
        );
        check_eq!(
            actual.samples.get(&sample_key(5, 1.5)),
            Some(&(hex(b"animated"), 0))
        );
        stream.caught_up(created.object_id, 3).await?;
        Ok(())
    })
}

fn test_captured_traffic_catchup_preserves_client_state(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = &ctx.session;
        let rows = capture();
        // Same replay selection as the acceptance oracle: the last initial
        // CreateLayer and all following edits, without the earlier overwritten generation.
        let start = rows
            .iter()
            .rposition(|row| row.kind == "CreateLayer")
            .ok_or_else(|| CaseError::new("The captured traffic holds no CreateLayer"))?;
        let created = client.create(session, &rows[start].payload()?).await?;
        let updates = &rows[start + 1..];
        check!(updates.len() >= 10);
        let midpoint = updates.len() / 2;
        let mut baseline: Option<(Scene, u64)> = None;
        let mut last: Option<proto::UpdateLayerResponse> = None;
        for (index, row) in (1..).zip(updates) {
            check_eq!(row.kind, "UpdateLayer");
            let response = client
                .update(session, created.object_id, &row.payload()?)
                .await?;
            if index == midpoint {
                let mut stream = client.subscribe(session, 0, 0).await?;
                let (_, payload) = stream.delivery().await?;
                let scene = apply_snapshot(&snapshot(&payload, 0)?, None)?;
                let baseline_seq = response.sequence_num;
                stream.caught_up(created.object_id, baseline_seq).await?;
                stream.close();
                baseline = Some((scene, baseline_seq));
            }
            last = Some(response);
        }
        let Some((baseline, baseline_seq)) = baseline else {
            return Err(CaseError::new("No baseline was read at the midpoint"));
        };
        check!(!baseline.sections.is_empty() && !baseline.fields.is_empty());
        // At least ten updates ran, so there is a last response.
        let response = last.ok_or_else(|| CaseError::new("No update ran"))?;
        let mut full = client.subscribe(session, 0, 0).await?;
        let (_, payload) = full.delivery().await?;
        let full_signature = snapshot(&payload, 0)?;
        let expected = apply_snapshot(&full_signature, None)?;
        let golden = capture_golden();
        check_eq!(field_order(&full_signature), field_order(&golden));
        check_eq!(
            expected,
            apply_snapshot(&golden, None)?,
            "Final scene differs from the pinned C++ reference scene"
        );
        full.caught_up(created.object_id, response.sequence_num)
            .await?;
        let mut reconnect = client
            .subscribe(session, created.object_id, baseline_seq)
            .await?;
        let (header, diff) = reconnect.delivery().await?;
        let signature = snapshot(&diff, baseline_seq)?;
        check_eq!(signature.base_version, baseline_seq);
        check_eq!(apply_snapshot(&signature, Some(&baseline))?, expected);
        check_eq!(header.sequence_num, response.sequence_num);
        reconnect
            .caught_up(created.object_id, response.sequence_num)
            .await?;
        Ok(())
    })
}

fn test_raw_structural_edits_have_independent_expected_state(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = &ctx.session;
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        client
            .update(session, created.object_id, fixture(Fixture::Structure))
            .await?;
        let mut stream = client.subscribe(session, 0, 0).await?;
        let (_, payload) = stream.delivery().await?;
        let signature = snapshot(&payload, 0)?;
        check_eq!(signature.source_format, "usda");
        let Scene {
            sections,
            fields,
            samples,
        } = apply_snapshot(&signature, None)?;
        check_eq!(
            sections,
            BTreeMap::from([
                (1, (0, String::new(), 0, 0)),
                (5, (1, "probe".to_string(), 0, 1)),
                (6, (1, "second".to_string(), 0, 0)),
                (7, (6, "moved".to_string(), 0, 0)),
            ])
        );
        check_eq!(
            fields,
            BTreeMap::from([((5, "value".to_string()), (hex(b"initial"), 0))])
        );
        check!(samples.is_empty());
        stream.caught_up(created.object_id, 2).await?;
        Ok(())
    })
}

/// The sibling-name rules, through one structural update and a fresh
/// catch-up, against a hand-written expected scene. Siblings collide only in
/// one slot: one name, and an attribute or a relationship both, or else one
/// `SdfSpecType`. Under the new prim `P` (10): a prim `x` (11) and an
/// attribute `x` (12) both stay; a relationship `x` (13) deletes the
/// attribute; renaming the prim `x` to `y` deletes the prim `y` (14) with its
/// child (15) first; renaming it `y` again changes nothing. A service that
/// keys collisions on the name alone deletes the prim `x` at the attribute's
/// create, and one whose rename deletes nothing keeps 14 and 15.
fn test_sibling_name_rules_have_independent_expected_state(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = &ctx.session;
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        client
            .update(session, created.object_id, fixture(Fixture::SiblingNames))
            .await?;
        let mut stream = client.subscribe(session, 0, 0).await?;
        let (_, payload) = stream.delivery().await?;
        let Scene {
            sections,
            fields,
            samples,
        } = apply_snapshot(&snapshot(&payload, 0)?, None)?;
        // (parent, name, `SdfSpecType`, position among the parent's
        // children).
        check_eq!(
            sections,
            BTreeMap::from([
                (1, (0, String::new(), 0, 0)),
                (5, (1, "probe".to_string(), 0, 0)),
                (10, (1, "P".to_string(), 6, 1)),
                (11, (10, "y".to_string(), 6, 0)),
                (13, (10, "x".to_string(), 8, 1)),
            ])
        );
        check_eq!(
            fields,
            BTreeMap::from([((5, "value".to_string()), (hex(b"initial"), 0))])
        );
        check!(samples.is_empty());
        stream.caught_up(created.object_id, 2).await?;
        Ok(())
    })
}

/// The name of a JoinSession message's `response_type` (Python's
/// `WhichOneof("response_type")`): `connected`, `peer_joined`, `peer_left`,
/// `peer_message`, or `none` when it carries none.
fn which(message: &proto::JoinSessionResponse) -> &'static str {
    match &message.response_type {
        Some(ResponseType::Connected(_)) => "connected",
        Some(ResponseType::PeerJoined(_)) => "peer_joined",
        Some(ResponseType::PeerLeft(_)) => "peer_left",
        Some(ResponseType::PeerMessage(_)) => "peer_message",
        None => "none",
    }
}

/// The `connection_id` of a JoinSession message of any `response_type`.
fn connection_id(message: &proto::JoinSessionResponse) -> Option<&str> {
    match &message.response_type {
        Some(ResponseType::Connected(event)) => Some(&event.connection_id),
        Some(ResponseType::PeerJoined(event)) => Some(&event.connection_id),
        Some(ResponseType::PeerLeft(event)) => Some(&event.connection_id),
        Some(ResponseType::PeerMessage(event)) => Some(&event.connection_id),
        None => None,
    }
}

/// The failure of a JoinSession message that is not the `expected` kind.
fn unexpected(expected: &str, message: &proto::JoinSessionResponse) -> CaseError {
    CaseError::new(format!(
        "Expected a {expected} message, got {}",
        which(message)
    ))
}

/// The `SessionConnected` of `message`, or a failure naming its kind.
fn connected(message: proto::JoinSessionResponse) -> CaseResult<proto::SessionConnected> {
    match message.response_type {
        Some(ResponseType::Connected(event)) => Ok(event),
        _ => Err(unexpected("connected", &message)),
    }
}

/// The `PeerJoined` of `message`, or a failure naming its kind.
fn peer_joined(message: proto::JoinSessionResponse) -> CaseResult<proto::PeerJoined> {
    match message.response_type {
        Some(ResponseType::PeerJoined(event)) => Ok(event),
        _ => Err(unexpected("peer_joined", &message)),
    }
}

/// The `PeerLeft` of `message`, or a failure naming its kind.
fn peer_left(message: proto::JoinSessionResponse) -> CaseResult<proto::PeerLeft> {
    match message.response_type {
        Some(ResponseType::PeerLeft(event)) => Ok(event),
        _ => Err(unexpected("peer_left", &message)),
    }
}

/// The `PeerMessage` of `message`, or a failure naming its kind.
fn peer_message(message: proto::JoinSessionResponse) -> CaseResult<proto::PeerMessage> {
    match message.response_type {
        Some(ResponseType::PeerMessage(event)) => Ok(event),
        _ => Err(unexpected("peer_message", &message)),
    }
}

fn test_presence_join_message_leave(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = &ctx.session;
        let (mut first, _first_send) = client.join(session, false).await?;
        let first_id = connected(first.next().await?)?.connection_id;
        check!(!first_id.is_empty());
        let (mut second, send) = client.join(session, false).await?;
        let second_id = connected(second.next().await?)?.connection_id;
        check!(!second_id.is_empty() && second_id != first_id);
        let joined = peer_joined(first.next().await?)?;
        check_eq!(joined.connection_id, second_id);
        check_eq!(joined.user_application, "conformance");
        send.message(b"conformance-message")?;
        let message = peer_message(first.next().await?)?;
        check_eq!(message.connection_id, second_id);
        check_eq!(message.data, b"conformance-message");
        second.close();
        let left = peer_left(first.next().await?)?;
        check_eq!(left.connection_id, second_id);
        Ok(())
    })
}

// The JoinSession client keepalive: a join that sets `client_keepalive` is told
// a non-zero `keepalive_interval_ms` and, once it sends nothing for three
// intervals, is ended `UNAVAILABLE` after its participant has left.

/// Slack on a silent join's ending beyond its three intervals.
const KEEPALIVE_ENDING_SLACK: Duration = Duration::from_secs(10);

/// The first message of a join, which must be its `SessionConnected`.
async fn connected_of(
    stream: &mut Stream<proto::JoinSessionResponse>,
) -> CaseResult<proto::SessionConnected> {
    let message = stream.next().await?;
    check_eq!(
        which(&message),
        "connected",
        "Expected SessionConnected first"
    );
    connected(message)
}

fn test_presence_keepalive_is_advertised_on_opt_in_and_never_delivered(
    ctx: &mut Ctx,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = &ctx.session;
        let (mut watcher, _watcher_send) = client.join(session, false).await?;
        check_eq!(
            connected_of(&mut watcher).await?.keepalive_interval_ms,
            0,
            "A join that did not opt in must be told keepalive interval 0"
        );
        let (mut sender, send) = client.join(session, true).await?;
        let connected = connected_of(&mut sender).await?;
        check!(
            connected.keepalive_interval_ms > 0,
            "A join that opted in must be told a non-zero keepalive interval"
        );
        let joined = peer_joined(watcher.next().await?)?;
        check_eq!(joined.connection_id, connected.connection_id);
        for _ in 0..5 {
            send.keepalive()?;
        }
        send.message(b"after-keepalives")?;
        let message = watcher.next().await?;
        check_eq!(
            which(&message),
            "peer_message",
            "Keepalives must never be delivered to peers"
        );
        let message = peer_message(message)?;
        check_eq!(message.connection_id, connected.connection_id);
        check_eq!(message.data, b"after-keepalives");
        Ok(())
    })
}

fn test_presence_silent_keepalive_join_ends_unavailable_after_leaving(
    ctx: &mut Ctx,
) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = &ctx.session;
        let (mut watcher, _watcher_send) = client.join(session, false).await?;
        connected_of(&mut watcher).await?;
        // The silent join's only send, its header, is after this instant.
        let opened = Instant::now();
        let (mut silent, _silent_send) = client.join(session, true).await?;
        let connected = connected_of(&mut silent).await?;
        let deadline = Duration::from_millis(connected.keepalive_interval_ms.saturating_mul(3));
        check!(
            deadline > Duration::ZERO,
            "A join that opted in must be told a non-zero keepalive interval"
        );
        let replayed = silent.next().await?;
        check_eq!(which(&replayed), "peer_joined");
        let replay_done = Instant::now();
        let result = silent
            .next_until(replay_done + deadline + KEEPALIVE_ENDING_SLACK)
            .await;
        let ended = Instant::now();
        expect_code!(result, Code::Unavailable);
        check!(
            ended - opened >= deadline,
            "Ended {:.1} s after the join's last send, before three intervals",
            (ended - opened).as_secs_f64()
        );
        let joined = peer_joined(watcher.next().await?)?;
        check_eq!(joined.connection_id, connected.connection_id);
        let left = peer_left(watcher.next().await?)?;
        check_eq!(left.connection_id, connected.connection_id);
        // A later join and its message fence the watcher's stream: no second
        // PeerLeft for the silent join arrives before them.
        let (mut fence, send) = client.join(session, false).await?;
        let fence_id = connected_of(&mut fence).await?.connection_id;
        send.message(b"fence")?;
        for kind in ["peer_joined", "peer_message"] {
            let event = watcher.next().await?;
            check_eq!(
                which(&event),
                kind,
                "Expected the fence's {kind}, got {}",
                which(&event)
            );
            check_eq!(connection_id(&event), Some(fence_id.as_str()));
        }
        Ok(())
    })
}
