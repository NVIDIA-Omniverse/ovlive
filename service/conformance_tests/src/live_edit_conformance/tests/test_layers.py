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
import grpc
from nvidia.omniverse.live.v1alpha import live_service_pb2 as pb
import pytest

from live_edit_conformance.protocol import (
    RETRY_INFO_TYPE,
    apply_snapshot,
    decode,
    delta_of_children_list,
    delta_of_entries,
    error_info,
    fixture,
    pack_ompf,
    snapshot,
    status_details,
)

DOMAIN = "nvidia.omniverse.live.v1alpha"

# The size limits a delta is refused for: entries (struct commands, set fields
# and time samples) summed, and the ids one ReorderChildren list names.
MAX_DELTA_ENTRIES = 99_999
MAX_CHILDREN_LIST_LEN = 100_000


def assert_size_refusal(error, reason, metadata):
    """INVALID_ARGUMENT carrying exactly one ErrorInfo `reason` in the service's
    domain with `metadata`, each value an unsigned decimal, and no RetryInfo:
    the refusal applied nothing and no re-send of the same delta can succeed."""
    assert error.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert error_info(error) == (reason, DOMAIN, metadata)
    assert all(value.isdigit() for value in metadata.values())
    assert RETRY_INFO_TYPE not in status_details(error), "A size refusal carries no RetryInfo"


def assert_delta_too_large(error, struct_commands, set_fields, time_samples):
    metadata = {
        "struct_commands": str(struct_commands),
        "set_fields": str(set_fields),
        "time_samples": str(time_samples),
        "max_entries": str(MAX_DELTA_ENTRIES),
    }
    assert_size_refusal(error, "DELTA_TOO_LARGE", metadata)
    message = error.details()
    total = struct_commands + set_fields + time_samples
    assert str(total) in message and str(MAX_DELTA_ENTRIES) in message, message
    assert "split" in message, "The message names the remedy"


def assert_children_list_too_long(error, actual):
    metadata = {"limit": str(MAX_CHILDREN_LIST_LEN), "actual": str(actual)}
    assert_size_refusal(error, "CHILDREN_LIST_TOO_LONG", metadata)


def test_layer_lifecycle_and_generation_precondition(client, session):
    first = client.create(session, fixture("initial"))
    assert first.object_id > 0
    response = client.update(session, first.object_id, fixture("update"))
    assert response.sequence_num == 2
    assert response.server_timestamp_microseconds_since_epoch > 0
    metadata = client.call("StatLayer", pb.StatLayerRequest(session_id=session, layer_name="layer"))
    assert metadata.created_at > 0 and metadata.last_modified_at >= metadata.created_at
    second = client.create(session, fixture("initial"))
    assert second.object_id > first.object_id
    with pytest.raises(grpc.RpcError) as error:
        client.update(session, first.object_id, fixture("update"))
    assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert client.update(session, second.object_id, b"").sequence_num == 2
    client.call("DeleteLayer", pb.DeleteLayerRequest(session_id=session, layer_name="layer"))
    with pytest.raises(grpc.RpcError) as error:
        client.call("StatLayer", pb.StatLayerRequest(session_id=session, layer_name="layer"))
    assert error.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.parametrize("payload", [b"not-ompf", b"OMPF\x01\x00\x00\x00", b"OMPF" + b"\xff" * 40])
def test_malformed_content_is_rejected_without_creating_layer(client, session, payload):
    with pytest.raises(grpc.RpcError) as error:
        client.create(session, payload)
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    with pytest.raises(grpc.RpcError) as error:
        client.call("StatLayer", pb.StatLayerRequest(session_id=session, layer_name="layer"))
    assert error.value.code() == grpc.StatusCode.NOT_FOUND


def test_delta_over_the_entry_limit_is_refused_delta_too_large_by_both_rpcs(client, session):
    # 60,000 struct commands + 60,000 set fields: each under 100,000, the sum
    # over 99,999. Built here, not a committed fixture.
    payload = pack_ompf(delta_of_entries(60_000, 60_000))
    with pytest.raises(grpc.RpcError) as error:
        client.create(session, payload, layer="oversized")
    assert_delta_too_large(error.value, 60_000, 60_000, 0)
    with pytest.raises(grpc.RpcError) as error:
        client.call("StatLayer", pb.StatLayerRequest(session_id=session, layer_name="oversized"))
    assert error.value.code() == grpc.StatusCode.NOT_FOUND, "No layer was created"

    created = client.create(session, fixture("initial"))
    with pytest.raises(grpc.RpcError) as error:
        client.update(session, created.object_id, payload)
    assert_delta_too_large(error.value, 60_000, 60_000, 0)
    # Nothing applied: the next commit is the layer's second.
    assert client.update(session, created.object_id, b"").sequence_num == 2


def test_delta_one_entry_over_the_limit_is_refused_and_one_at_it_never_delta_too_large(client, session):
    created = client.create(session, fixture("initial"))
    with pytest.raises(grpc.RpcError) as error:
        client.update(session, created.object_id, pack_ompf(delta_of_entries(0, 100_000)))
    assert_delta_too_large(error.value, 0, 100_000, 0)
    # Exactly the limit is never refused DELTA_TOO_LARGE. These entries'
    # content may be accepted or refused on its own merits; this checks only
    # that a refusal, if any, does not carry that reason.
    try:
        client.update(session, created.object_id, pack_ompf(delta_of_entries(0, 99_999)))
    except grpc.RpcError as rejected:
        assert error_info(rejected) is None or error_info(rejected)[0] != "DELTA_TOO_LARGE"


@pytest.mark.parametrize("children", [MAX_CHILDREN_LIST_LEN + 1, 1_000_001])
def test_children_list_over_the_limit_is_refused_children_list_too_long_by_both_rpcs(
    client, session, children
):
    payload = pack_ompf(delta_of_children_list(children))
    with pytest.raises(grpc.RpcError) as error:
        client.create(session, payload, layer="oversized")
    assert_children_list_too_long(error.value, children)
    created = client.create(session, fixture("initial"))
    with pytest.raises(grpc.RpcError) as error:
        client.update(session, created.object_id, payload)
    assert_children_list_too_long(error.value, children)
    assert client.update(session, created.object_id, b"").sequence_num == 2


@pytest.mark.parametrize("operation", ["CreateLayer", "UpdateLayer"])
def test_chunk_before_header_is_invalid(client, session, operation):
    request_type = getattr(pb, f"{operation}Request")
    chunk_type = getattr(pb, f"{operation}Chunk")
    with pytest.raises(grpc.RpcError) as error:
        client.call(operation, iter([request_type(chunk=chunk_type(data=fixture("initial")))]))
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.parametrize("name", ["initial", "time_sample", "external", "large"])
def test_content_round_trip(client, session, settings, name):
    payload = fixture(name)
    created = client.create(session, payload, chunk_size=4096)
    stream = client.subscribe(session)
    header, returned = stream.delivery()
    assert (header.object_id, header.sequence_num) == (created.object_id, 1)
    sent_signature = decode(settings, payload)
    received_signature = snapshot(settings, returned)
    assert received_signature["sourceFormat"] == sent_signature["sourceFormat"]
    expected = apply_snapshot(sent_signature)
    actual = apply_snapshot(received_signature)
    assert actual == expected
    stream.caught_up(created.object_id, 1)


def test_missing_layer_subscription_returns_not_found(client, session):
    stream = client.subscribe(session)
    with pytest.raises(grpc.RpcError) as error:
        stream.next()
    assert error.value.code() == grpc.StatusCode.NOT_FOUND


# A client operation id: the recommended UUIDv4 text form.
OPERATION_ID = "3f2b6c1e-8d4a-4b7e-9c05-1a2b3c4d5e6f"


def test_requests_without_operation_id_are_answered_unspecified(client, session):
    created = client.create(session, fixture("initial"))
    assert created.operation_outcome == pb.OPERATION_OUTCOME_UNSPECIFIED
    updated = client.update(session, created.object_id, fixture("update"))
    assert updated.operation_outcome == pb.OPERATION_OUTCOME_UNSPECIFIED


def test_update_resend_with_operation_id_is_answered_already_committed(client, session):
    created = client.create(session, fixture("initial"))
    first = client.update(
        session, created.object_id, fixture("update"), timestamp=111, operation_id=OPERATION_ID
    )
    assert first.operation_outcome == pb.OPERATION_OUTCOME_APPLIED
    # A re-send whose response was lost: same id and bytes, fresh timestamp.
    again = client.update(
        session, created.object_id, fixture("update"), timestamp=222, operation_id=OPERATION_ID
    )
    assert again.operation_outcome == pb.OPERATION_OUTCOME_ALREADY_COMMITTED
    assert again.sequence_num == first.sequence_num
    assert (
        again.server_timestamp_microseconds_since_epoch
        == first.server_timestamp_microseconds_since_epoch
    )
    # The re-send applied nothing: the next commit is the very next sequence.
    assert client.update(session, created.object_id, b"").sequence_num == first.sequence_num + 1


def test_create_resend_with_operation_id_is_answered_already_committed(client, session):
    first = client.create(session, fixture("initial"), operation_id=OPERATION_ID)
    assert first.operation_outcome == pb.OPERATION_OUTCOME_APPLIED
    again = client.create(session, fixture("initial"), operation_id=OPERATION_ID)
    assert again.operation_outcome == pb.OPERATION_OUTCOME_ALREADY_COMMITTED
    assert again.object_id == first.object_id
    # The layer was not replaced by its own re-send: the original object takes writes.
    assert client.update(session, first.object_id, b"").sequence_num == 2


def test_operation_id_reused_for_a_different_request_is_already_exists(client, session):
    created = client.create(session, fixture("initial"), operation_id=OPERATION_ID)
    with pytest.raises(grpc.RpcError) as error:
        client.create(session, fixture("time_sample"), operation_id=OPERATION_ID)
    assert error.value.code() == grpc.StatusCode.ALREADY_EXISTS
    client.update(session, created.object_id, fixture("update"), operation_id=OPERATION_ID)
    with pytest.raises(grpc.RpcError) as error:
        client.update(session, created.object_id, b"", operation_id=OPERATION_ID)
    assert error.value.code() == grpc.StatusCode.ALREADY_EXISTS
    # Neither refusal changed the layer.
    assert client.update(session, created.object_id, b"").sequence_num == 3


def test_update_operation_id_without_client_id_is_invalid_argument(client, session):
    created = client.create(session, fixture("initial"), operation_id=OPERATION_ID)
    assert created.operation_outcome == pb.OPERATION_OUTCOME_APPLIED
    # An operation id is guarded per (object, client_id): it needs a client id.
    with pytest.raises(grpc.RpcError) as error:
        client.update(
            session, created.object_id, fixture("update"), client_id="", operation_id=OPERATION_ID
        )
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    # Refused before anything was applied; an empty client id without an
    # operation id is still accepted.
    assert client.update(session, created.object_id, b"", client_id="").sequence_num == 2


@pytest.mark.parametrize("operation_id", ["a" * 65, "op:1", "op 1"])
@pytest.mark.parametrize("operation", ["CreateLayer", "UpdateLayer"])
def test_malformed_operation_id_is_invalid_argument(client, session, operation, operation_id):
    created = client.create(session, fixture("initial"))
    with pytest.raises(grpc.RpcError) as error:
        if operation == "CreateLayer":
            client.create(session, fixture("initial"), operation_id=operation_id)
        else:
            client.update(session, created.object_id, fixture("update"), operation_id=operation_id)
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    # Refused before anything was applied.
    assert client.update(session, created.object_id, b"").sequence_num == 2
