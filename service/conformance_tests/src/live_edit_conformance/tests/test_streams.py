# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import base64
import time

import grpc
from nvidia.omniverse.live.v1alpha import live_service_pb2 as pb
import pytest

from live_edit_conformance.client import KEEPALIVE
from live_edit_conformance.protocol import (
    apply_snapshot,
    capture,
    capture_golden,
    decode,
    fixture,
    snapshot,
    field_order,
)


def test_live_delivery_and_self_echo_suppression(client, session):
    created = client.create(session, fixture("initial"))
    stream = client.subscribe(session, created.object_id, 1, client_id="self")
    stream.caught_up(created.object_id, 1)
    assert (
        client.update(session, created.object_id, fixture("update"), client_id="self").sequence_num
        == 2
    )
    # A second writer's commit is an ordering barrier: a broken self-echo
    # filter delivers seq 2 before seq 3 and fails, without a silence sleep.
    response = client.update(session, created.object_id, fixture("update"), client_id="other")
    header, payload = stream.delivery()
    assert header.sequence_num == response.sequence_num == 3
    assert header.object_id == created.object_id
    assert header.client_timestamp_microseconds_since_epoch == 123456
    assert (
        header.server_timestamp_microseconds_since_epoch
        == response.server_timestamp_microseconds_since_epoch
    )
    assert payload == fixture("update"), "Steady-state delivery must preserve the writer's OMPF"


def test_empty_update_is_delivered_with_commit_timestamp(client, session, settings):
    created = client.create(session, fixture("initial"))
    stream = client.subscribe(session, created.object_id, 1)
    stream.caught_up(created.object_id, 1)
    response = client.update(session, created.object_id, b"")
    header, payload = stream.delivery()
    assert header.sequence_num == response.sequence_num == 2
    if payload:
        signature = decode(settings, payload)
        assert (
            not signature["structCommands"]
            and not signature["setFields"]
            and not signature["timeSamples"]
        )
    assert (
        header.server_timestamp_microseconds_since_epoch
        == response.server_timestamp_microseconds_since_epoch
    )


def test_overwrite_rebootstraps_subscription(client, session, settings):
    first = client.create(session, fixture("initial"))
    stream = client.subscribe(session, first.object_id, 1)
    stream.caught_up(first.object_id, 1)
    second = client.create(session, fixture("time_sample"))
    header, payload = stream.delivery()
    assert header.object_id == second.object_id != first.object_id
    assert header.sequence_num == 1
    assert apply_snapshot(snapshot(settings, payload)) == apply_snapshot(
        decode(settings, fixture("time_sample"))
    )
    stream.caught_up(second.object_id, 1)
    client.call("DeleteLayer", pb.DeleteLayerRequest(session_id=session, layer_name="layer"))
    with pytest.raises(grpc.RpcError) as error:
        stream.next()
    assert error.value.code() == grpc.StatusCode.NOT_FOUND


def test_reconnect_folds_fields_and_time_samples(client, session, settings):
    created = client.create(session, fixture("initial"))
    baseline_stream = client.subscribe(session)
    _, baseline_payload = baseline_stream.delivery()
    baseline_stream.caught_up(created.object_id, 1)
    baseline_stream.close()
    client.update(session, created.object_id, fixture("update"))
    response = client.update(session, created.object_id, fixture("update_time"))
    stream = client.subscribe(session, created.object_id, 1)
    header, payload = stream.delivery()
    assert header.sequence_num == response.sequence_num == 3
    signature = snapshot(settings, payload, base_version=1)
    assert signature["isDiff"] and signature["baseVersion"] == 1
    baseline = apply_snapshot(snapshot(settings, baseline_payload))
    actual = apply_snapshot(signature, baseline)
    # Fixed expected values, independent of a second server read.
    assert actual[1][(5, "value")] == (b"updated".hex(), 0)
    assert actual[2][(5, 1.5)] == (b"animated".hex(), 0)
    stream.caught_up(created.object_id, 3)


def test_captured_traffic_catchup_preserves_client_state(client, session, settings):
    rows = capture()
    # Same replay selection as the acceptance oracle: the last initial
    # CreateLayer and all following edits, without the earlier overwritten generation.
    start = max(i for i, row in enumerate(rows) if row["kind"] == "CreateLayer")
    created = client.create(session, base64.b64decode(rows[start]["b64"]))
    updates = rows[start + 1 :]
    assert len(updates) >= 10
    midpoint = len(updates) // 2
    baseline = None
    for index, row in enumerate(updates, 1):
        assert row["kind"] == "UpdateLayer"
        response = client.update(session, created.object_id, base64.b64decode(row["b64"]))
        if index == midpoint:
            stream = client.subscribe(session)
            _, payload = stream.delivery()
            baseline = apply_snapshot(snapshot(settings, payload))
            baseline_seq = response.sequence_num
            stream.caught_up(created.object_id, baseline_seq)
            stream.close()
    assert baseline is not None and baseline[0] and baseline[1]
    full = client.subscribe(session)
    _, payload = full.delivery()
    full_signature = snapshot(settings, payload)
    expected = apply_snapshot(full_signature)
    assert field_order(full_signature) == field_order(capture_golden())
    assert expected == apply_snapshot(capture_golden()), (
        "Final scene differs from the pinned C++ oracle"
    )
    full.caught_up(created.object_id, response.sequence_num)
    reconnect = client.subscribe(session, created.object_id, baseline_seq)
    header, diff = reconnect.delivery()
    signature = snapshot(settings, diff, base_version=baseline_seq)
    assert signature["baseVersion"] == baseline_seq
    assert apply_snapshot(signature, baseline) == expected
    assert header.sequence_num == response.sequence_num
    reconnect.caught_up(created.object_id, response.sequence_num)


def test_raw_structural_edits_have_independent_expected_state(client, session, settings):
    created = client.create(session, fixture("initial"))
    client.update(session, created.object_id, fixture("structure"))
    stream = client.subscribe(session)
    _, payload = stream.delivery()
    signature = snapshot(settings, payload)
    assert signature["sourceFormat"] == "usda"
    sections, fields, samples = apply_snapshot(signature)
    assert sections == {
        1: (0, "", 0, 0),
        5: (1, "probe", 0, 1),
        6: (1, "second", 0, 0),
        7: (6, "moved", 0, 0),
    }
    assert fields == {(5, "value"): (b"initial".hex(), 0)} and not samples
    stream.caught_up(created.object_id, 2)


def test_presence_join_message_leave(client, session):
    first, _ = client.join(session)
    connected = first.next()
    assert connected.WhichOneof("response_type") == "connected"
    first_id = connected.connected.connection_id
    assert first_id
    second, send = client.join(session)
    connected = second.next()
    assert connected.WhichOneof("response_type") == "connected"
    second_id = connected.connected.connection_id
    assert second_id and second_id != first_id
    joined = first.next()
    assert joined.WhichOneof("response_type") == "peer_joined"
    assert joined.peer_joined.connection_id == second_id
    assert joined.peer_joined.user_application == "conformance"
    send.put(b"conformance-message")
    message = first.next()
    assert message.WhichOneof("response_type") == "peer_message"
    assert message.peer_message.connection_id == second_id
    assert message.peer_message.data == b"conformance-message"
    second.close()
    left = first.next()
    assert left.WhichOneof("response_type") == "peer_left"
    assert left.peer_left.connection_id == second_id


# The JoinSession client keepalive: a join that sets `client_keepalive` is told
# a non-zero `keepalive_interval_ms` and, once it sends nothing for three
# intervals, is ended `UNAVAILABLE` after its participant has left.

# Slack on a silent join's ending beyond its three intervals.
KEEPALIVE_ENDING_SLACK = 10


def connected_of(stream):
    message = stream.next()
    assert message.WhichOneof("response_type") == "connected", "Expected SessionConnected first"
    return message.connected


def test_presence_keepalive_is_advertised_on_opt_in_and_never_delivered(client, session):
    watcher, _ = client.join(session)
    assert connected_of(watcher).keepalive_interval_ms == 0, (
        "A join that did not opt in must be told keepalive interval 0"
    )
    sender, send = client.join(session, keepalive=True)
    connected = connected_of(sender)
    assert connected.keepalive_interval_ms > 0, (
        "A join that opted in must be told a non-zero keepalive interval"
    )
    joined = watcher.next()
    assert joined.WhichOneof("response_type") == "peer_joined"
    assert joined.peer_joined.connection_id == connected.connection_id
    for _ in range(5):
        send.put(KEEPALIVE)
    send.put(b"after-keepalives")
    message = watcher.next()
    assert message.WhichOneof("response_type") == "peer_message", (
        "Keepalives must never be delivered to peers"
    )
    assert message.peer_message.connection_id == connected.connection_id
    assert message.peer_message.data == b"after-keepalives"


def test_presence_silent_keepalive_join_ends_unavailable_after_leaving(client, session):
    watcher, _ = client.join(session)
    connected_of(watcher)
    # The silent join's only send, its header, is after this instant.
    opened = time.monotonic()
    silent, _ = client.join(session, keepalive=True)
    connected = connected_of(silent)
    deadline = 3 * connected.keepalive_interval_ms / 1000
    assert deadline > 0, "A join that opted in must be told a non-zero keepalive interval"
    replayed = silent.next()
    assert replayed.WhichOneof("response_type") == "peer_joined"
    replay_done = time.monotonic()
    with pytest.raises(grpc.RpcError) as error:
        silent.next(replay_done + deadline + KEEPALIVE_ENDING_SLACK)
    ended = time.monotonic()
    assert error.value.code() == grpc.StatusCode.UNAVAILABLE
    assert ended - opened >= deadline, (
        f"Ended {ended - opened:.1f} s after the join's last send, before three intervals"
    )
    joined = watcher.next()
    assert joined.WhichOneof("response_type") == "peer_joined"
    assert joined.peer_joined.connection_id == connected.connection_id
    left = watcher.next()
    assert left.WhichOneof("response_type") == "peer_left"
    assert left.peer_left.connection_id == connected.connection_id
    # A later join and its message fence the watcher's stream: no second
    # PeerLeft for the silent join arrives before them.
    fence, send = client.join(session)
    fence_id = connected_of(fence).connection_id
    send.put(b"fence")
    for kind in ("peer_joined", "peer_message"):
        event = watcher.next()
        assert event.WhichOneof("response_type") == kind, (
            f"Expected the fence's {kind}, got {event.WhichOneof('response_type')}"
        )
        assert getattr(event, kind).connection_id == fence_id
