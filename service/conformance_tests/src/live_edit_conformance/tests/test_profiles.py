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
from grpc_health.v1 import health_pb2, health_pb2_grpc
from nvidia.omniverse.live.v1alpha import live_service_pb2 as pb
import pytest

from live_edit_conformance.client import Client
from live_edit_conformance.protocol import fixture


@pytest.mark.deployment
def test_live_service_health_is_serving_without_auth(client):
    response = health_pb2_grpc.HealthStub(client.channel).Check(
        health_pb2.HealthCheckRequest(service="nvidia.omniverse.live.v1alpha.LiveService"),
        timeout=client.settings.timeout,
    )
    assert response.status == health_pb2.HealthCheckResponse.SERVING


@pytest.mark.permissions
def test_denied_base_path_returns_permission_denied(client):
    with pytest.raises(grpc.RpcError) as error:
        client.call("ListSessions", pb.ListSessionsRequest(path=client.settings.denied_path))
    assert error.value.code() == grpc.StatusCode.PERMISSION_DENIED


@pytest.mark.multi_replica
def test_update_across_replica_endpoints(client, session, settings):
    peer = Client(settings, settings.peer_endpoint)
    try:
        created = client.create(session, fixture("initial"))
        stream = peer.subscribe(session, created.object_id, 1)
        stream.caught_up(created.object_id, 1)
        response = client.update(session, created.object_id, fixture("update"))
        header, payload = stream.delivery()
        assert header.object_id == created.object_id
        assert header.sequence_num == response.sequence_num == 2
        assert payload == fixture("update")
    finally:
        peer.close()
