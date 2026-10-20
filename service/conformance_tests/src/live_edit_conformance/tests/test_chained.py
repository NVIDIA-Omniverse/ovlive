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

"""Chained UpdateLayer writes (UpdateLayerRequest.chained_header).

A chained call names its predecessor, the same client_id's previous write to
the object, and commits only directly after it, at most once. Every case runs
one writer's chain on a fresh layer; sequence numbers prove the commit order.
"""

import time

import grpc
from nvidia.omniverse.live.v1alpha import live_service_pb2 as pb
import pytest

from live_edit_conformance.client import Client
from live_edit_conformance.protocol import (
    RETRY_INFO_TYPE,
    delta_of_children_list,
    delta_of_entries,
    error_info,
    fixture,
    pack_ompf,
    status_details,
)

DOMAIN = "nvidia.omniverse.live.v1alpha"

# How long a call sent ahead of its predecessor is watched to still be
# waiting: the observation itself, not a synchronization (an implementation
# that commits it at once fails here).
STILL_WAITING = 0.3


def ops(prefix, count):
    """`count` chain operation ids: UUID-free but unique to the case's fresh layer."""
    return [f"{prefix}-{i}" for i in range(1, count + 1)]


def assert_applied(response, sequence):
    assert response.operation_outcome == pb.OPERATION_OUTCOME_APPLIED
    assert response.sequence_num == sequence


def assert_still_waiting(*futures):
    time.sleep(STILL_WAITING)
    for future in futures:
        assert not future.done(), "A call whose predecessor has not committed must wait"


def assert_predecessor_not_committed(error, metadata):
    assert error.code() == grpc.StatusCode.ABORTED
    assert error_info(error) == (
        "PREDECESSOR_NOT_COMMITTED",
        DOMAIN,
        metadata,
    ), "ABORTED carries the predecessor refusal's ErrorInfo"
    assert RETRY_INFO_TYPE not in status_details(error), "The refusal carries no RetryInfo"


def test_ping_reports_operation_ids_and_chained_writes(client):
    response = client.call("Ping", pb.PingRequest(client_timestamp_microseconds_since_epoch=7))
    assert response.client_timestamp_microseconds_since_epoch == 7
    assert response.operation_ids is True
    assert response.chained_writes is True


def test_chained_writes_sent_in_order_commit_in_chain_order(client, session):
    created = client.create(session, fixture("initial"))
    chain = ops("in-order", 3)
    for i, operation_id in enumerate(chain):
        previous = chain[i - 1] if i else ""
        response = client.chained_update(
            session, created.object_id, fixture("update"), operation_id, previous
        )
        assert_applied(response, i + 2)


def test_chained_writes_sent_out_of_order_commit_in_chain_order(client, session):
    created = client.create(session, fixture("initial"))
    first, second, third = ops("out-of-order", 3)
    third_call = client.chained_update_async(
        session, created.object_id, fixture("update"), third, second
    )
    second_call = client.chained_update_async(
        session, created.object_id, fixture("update"), second, first
    )
    assert_still_waiting(third_call, second_call)
    assert_applied(
        client.chained_update(session, created.object_id, fixture("update"), first, ""), 2
    )
    assert_applied(second_call.result(), 3)
    assert_applied(third_call.result(), 4)


def test_chained_resend_of_the_latest_write_is_answered_already_committed(client, session):
    created = client.create(session, fixture("initial"))
    first, second = ops("resend", 2)
    client.chained_update(session, created.object_id, fixture("update"), first, "")
    committed = client.chained_update(session, created.object_id, fixture("update"), second, first)
    again = client.chained_update(
        session, created.object_id, fixture("update"), second, first, no_wait=True
    )
    assert again.operation_outcome == pb.OPERATION_OUTCOME_ALREADY_COMMITTED
    assert again.sequence_num == committed.sequence_num
    # The re-send applied nothing.
    assert client.update(session, created.object_id, b"").sequence_num == 4


def test_waiting_chained_call_is_refused_aborted_before_its_deadline(client, session):
    created = client.create(session, fixture("initial"))
    first, second, third = ops("deadline", 3)
    client.chained_update(session, created.object_id, fixture("update"), first, "")
    # `second` never commits: `third` waits until its margin before its deadline.
    with pytest.raises(grpc.RpcError) as error:
        client.chained_update_async(
            session, created.object_id, fixture("update"), third, second, timeout=3
        ).result()
    assert_predecessor_not_committed(
        error.value, {"last_record": "operation", "last_operation_id": first}
    )
    assert client.update(session, created.object_id, b"").sequence_num == 3


def test_no_wait_call_whose_predecessor_was_passed_is_refused_at_once(client, session):
    created = client.create(session, fixture("initial"))
    first, second, third = ops("passed", 3)
    # Staged with ordinary operation-id commits: the record has passed `first`.
    for operation_id in (first, second, third):
        client.update(session, created.object_id, fixture("update"), operation_id=operation_id)
    with pytest.raises(grpc.RpcError) as error:
        client.chained_update(
            session, created.object_id, fixture("update"), second, first, no_wait=True
        )
    assert_predecessor_not_committed(
        error.value, {"last_record": "operation", "last_operation_id": third}
    )
    with pytest.raises(grpc.RpcError) as error:
        client.chained_update(
            session,
            created.object_id,
            fixture("update"),
            "never",
            "unknown",
            client_id="other-writer",
            no_wait=True,
        )
    assert_predecessor_not_committed(error.value, {"last_record": "absent"})
    assert client.update(session, created.object_id, b"").sequence_num == 5


def test_cancelled_waiting_chained_call_applies_nothing(client, session):
    created = client.create(session, fixture("initial"))
    first, second = ops("cancel", 2)
    waiting = client.chained_update_async(
        session, created.object_id, fixture("update"), second, first
    )
    assert_still_waiting(waiting)
    assert waiting.cancel(), "The waiting call must still be cancellable"
    assert_applied(
        client.chained_update(session, created.object_id, fixture("update"), first, ""), 2
    )
    # Its own re-send applies: the cancelled call never committed.
    assert_applied(
        client.chained_update(
            session, created.object_id, fixture("update"), second, first, no_wait=True
        ),
        3,
    )


def test_predecessor_lost_before_its_stream_ended_and_resent_releases_its_successor(
    client, session
):
    created = client.create(session, fixture("initial"))
    first, second = ops("lost", 2)
    successor = client.chained_update_async(
        session, created.object_id, fixture("update"), second, first
    )
    lost = client.unended_update_async(
        client.chained_requests(session, created.object_id, fixture("update"), first, "")
    )
    assert_still_waiting(successor, lost)
    assert lost.cancel(), "The unended predecessor must still be cancellable"
    assert not successor.done(), "The successor waits for its predecessor's re-send"
    assert_applied(
        client.chained_update(session, created.object_id, fixture("update"), first, ""), 2
    )
    assert_applied(successor.result(), 3)


def test_invalid_chained_call_carries_invalid_update(client, session):
    created = client.create(session, fixture("initial"))
    with pytest.raises(grpc.RpcError) as error:
        client.chained_update(session, created.object_id, fixture("update"), "same", "same")
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert error_info(error.value) == ("INVALID_UPDATE", DOMAIN, {})
    # The unchained INVALID_ARGUMENT keeps no ErrorInfo.
    with pytest.raises(grpc.RpcError) as error:
        client.update(session, created.object_id, b"", operation_id="op 1")
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert error_info(error.value) is None
    assert client.update(session, created.object_id, b"").sequence_num == 2


def test_size_refused_chained_call_carries_its_size_reason_in_place_of_invalid_update(
    client, session
):
    created = client.create(session, fixture("initial"))
    # 60,000 struct commands + 60,000 set fields: over the 99,999-entry limit.
    with pytest.raises(grpc.RpcError) as error:
        client.chained_update(
            session, created.object_id, pack_ompf(delta_of_entries(60_000, 60_000)), "big", ""
        )
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert error_info(error.value) == (
        "DELTA_TOO_LARGE",
        DOMAIN,
        {
            "struct_commands": "60000",
            "set_fields": "60000",
            "time_samples": "0",
            "max_entries": "99999",
        },
    ), "A size refusal keeps its own reason on a chained call"
    assert RETRY_INFO_TYPE not in status_details(error.value)
    # A ReorderChildren list one id over the 100,000-id limit.
    with pytest.raises(grpc.RpcError) as error:
        client.chained_update(
            session, created.object_id, pack_ompf(delta_of_children_list(100_001)), "long", ""
        )
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert error_info(error.value) == (
        "CHILDREN_LIST_TOO_LONG",
        DOMAIN,
        {"limit": "100000", "actual": "100001"},
    )
    assert RETRY_INFO_TYPE not in status_details(error.value)
    assert client.update(session, created.object_id, b"").sequence_num == 2


@pytest.mark.multi_replica
def test_chained_writes_across_replica_endpoints_commit_in_chain_order(client, session, settings):
    peer = Client(settings, settings.peer_endpoint)
    try:
        created = client.create(session, fixture("initial"))
        first, second, third = ops("replicas", 3)
        third_call = peer.chained_update_async(
            session, created.object_id, fixture("update"), third, second
        )
        second_call = client.chained_update_async(
            session, created.object_id, fixture("update"), second, first
        )
        assert_still_waiting(third_call, second_call)
        assert_applied(
            peer.chained_update(session, created.object_id, fixture("update"), first, ""), 2
        )
        assert_applied(second_call.result(), 3)
        assert_applied(third_call.result(), 4)
    finally:
        peer.close()
