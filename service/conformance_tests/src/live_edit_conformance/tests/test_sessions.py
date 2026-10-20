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
import uuid

import grpc
from nvidia.omniverse.live.v1alpha import live_service_pb2 as pb
import pytest

from live_edit_conformance.protocol import fixture


def test_ping_echoes_timestamp(client):
    response = client.call("Ping", pb.PingRequest(client_timestamp_microseconds_since_epoch=12345))
    assert response.client_timestamp_microseconds_since_epoch == 12345
    assert response.server_timestamp_microseconds_since_epoch > 0


def test_session_create_list_delete_and_duplicate_name(client):
    name = f"conformance-{uuid.uuid4().hex}"
    session = client.session(name=name)
    rows = client.call(
        "ListSessions", pb.ListSessionsRequest(path=client.settings.base_path)
    ).sessions
    matching = [row for row in rows if row.session_id == session]
    assert len(matching) == 1 and matching[0].name == name
    assert matching[0].created_at > 0
    with pytest.raises(grpc.RpcError) as error:
        client.session(name=name)
    assert error.value.code() == grpc.StatusCode.ALREADY_EXISTS
    client.call("DeleteSession", pb.DeleteSessionRequest(session_id=session))
    rows = client.call(
        "ListSessions", pb.ListSessionsRequest(path=client.settings.base_path)
    ).sessions
    assert session not in {row.session_id for row in rows}


@pytest.mark.parametrize(
    "name,message",
    [
        ("CreateSession", pb.CreateSessionRequest()),
        ("ListSessions", pb.ListSessionsRequest()),
        ("DeleteSession", pb.DeleteSessionRequest()),
        ("StatLayer", pb.StatLayerRequest()),
        ("DeleteLayer", pb.DeleteLayerRequest()),
    ],
)
def test_empty_identifiers_are_invalid(client, name, message):
    with pytest.raises(grpc.RpcError) as error:
        client.call(name, message)
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_anonymous_request_is_rejected(client):
    with pytest.raises(grpc.RpcError) as error:
        client.call(
            "CreateSession",
            pb.CreateSessionRequest(
                path=client.settings.base_path, name=f"anonymous-{uuid.uuid4().hex}"
            ),
            metadata=[],
        )
    assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_sessions_isolate_layer_names(client, session):
    other = client.session()
    client.create(session, fixture("initial"))
    with pytest.raises(grpc.RpcError) as error:
        client.call("StatLayer", pb.StatLayerRequest(session_id=other, layer_name="layer"))
    assert error.value.code() == grpc.StatusCode.NOT_FOUND


def test_session_delete_cascades_to_layer_and_subscription(client, session):
    created = client.create(session, fixture("initial"))
    stream = client.subscribe(session, created.object_id, 1)
    stream.caught_up(created.object_id, 1)
    client.call("DeleteSession", pb.DeleteSessionRequest(session_id=session))
    with pytest.raises(grpc.RpcError) as error:
        stream.next()
    assert error.value.code() == grpc.StatusCode.NOT_FOUND
    with pytest.raises(grpc.RpcError) as error:
        client.call("StatLayer", pb.StatLayerRequest(session_id=session, layer_name="layer"))
    assert error.value.code() == grpc.StatusCode.NOT_FOUND
