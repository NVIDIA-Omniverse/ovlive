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

"""In-cluster smoke tests for the Live Edit gRPC service.

These tests verify wire connectivity, auth handshake, and the Valkey-backed
session path. They are deliberately minimal — they do not exercise OODF
delta construction or layer content writes (those would require shipping
prebuilt OMPF fixtures and an OODF builder in the smoke-test image).
"""

import logging
import time

import grpc
import pytest
from live_edit_helpers import create_session, delete_session, list_sessions
from utils import timeout_grpc_call

logger = logging.getLogger(__name__)


@pytest.mark.grpc
def test_ping(live_edit_stub, grpc_metadata):
    """Ping the service and verify the server echoes our client timestamp."""
    from nvidia.omniverse.live.v1alpha import live_service_pb2

    client_ts = int(time.time() * 1_000_000)
    req = live_service_pb2.PingRequest(
        client_timestamp_microseconds_since_epoch=client_ts,
    )
    resp = timeout_grpc_call(live_edit_stub.Ping, req, metadata=grpc_metadata)
    assert resp.client_timestamp_microseconds_since_epoch == client_ts
    assert resp.server_timestamp_microseconds_since_epoch > 0
    logger.info(
        "Ping succeeded - server_ts=%s",
        resp.server_timestamp_microseconds_since_epoch,
    )


@pytest.mark.grpc
def test_create_list_and_delete_session(
    live_edit_stub, grpc_metadata, base_path, session_name
):
    """Create a session, list sessions for the path, then delete it."""
    session_id = None
    try:
        session_id = create_session(
            live_edit_stub, grpc_metadata, base_path, session_name
        )
        logger.info("CreateSession succeeded - session_id=%s", session_id)

        resp = list_sessions(live_edit_stub, grpc_metadata, base_path)
        ids = {s.session_id for s in resp.sessions}
        assert session_id in ids, (
            f"ListSessions did not include freshly-created session_id "
            f"{session_id}; got {ids}"
        )
        names = {s.name for s in resp.sessions if s.session_id == session_id}
        assert session_name in names, (
            f"ListSessions returned session_id {session_id} with wrong name; "
            f"expected {session_name!r}, got {names!r}"
        )
        logger.info("ListSessions succeeded - %d sessions for path", len(ids))

        delete_session(live_edit_stub, grpc_metadata, session_id)
        logger.info("DeleteSession succeeded")
        session_id = None
    finally:
        if session_id:
            try:
                delete_session(live_edit_stub, grpc_metadata, session_id)
            except Exception:
                pass


@pytest.mark.grpc
def test_join_session_bidi_connects(
    live_edit_stub, grpc_metadata, base_path, session_name
):
    """Open a JoinSession bidi stream and verify SessionConnected arrives."""
    from nvidia.omniverse.live.v1alpha import live_service_pb2

    session_id = None
    try:
        session_id = create_session(
            live_edit_stub, grpc_metadata, base_path, session_name
        )

        header = live_service_pb2.JoinSessionHeader(
            session_id=session_id,
            user_application="live-edit-smoke-test",
        )
        first = live_service_pb2.JoinSessionRequest(header=header)

        def request_generator():
            yield first
            # Block until the server cancels the stream on session delete.
            # We never send a second message; just keep the iterator open
            # long enough to read the SessionConnected response.
            time.sleep(1)

        stream = timeout_grpc_call(
            live_edit_stub.JoinSession,
            request_generator(),
            metadata=grpc_metadata,
        )

        first_resp = next(stream)
        assert first_resp.WhichOneof("response_type") == "connected", (
            f"JoinSession first response should be SessionConnected, "
            f"got {first_resp.WhichOneof('response_type')!r}"
        )
        assert first_resp.connected.connection_id, (
            "SessionConnected.connection_id must not be empty"
        )
        logger.info(
            "JoinSession connected - connection_id=%s",
            first_resp.connected.connection_id,
        )
        stream.cancel()
    finally:
        if session_id:
            try:
                delete_session(live_edit_stub, grpc_metadata, session_id)
            except Exception:
                pass


@pytest.mark.grpc
def test_subscribe_read_layer_returns_not_found(
    live_edit_stub, grpc_metadata, base_path, session_name, client_id
):
    """SubscribeReadLayer on a non-existent layer must surface NOT_FOUND.

    We can't write a layer in a smoke test (OODF delta construction is
    out of scope), so the assertion is that the server fails the stream
    with NOT_FOUND for a layer that was never created. This proves the
    server-streaming RPC reaches the handler, validates the request, and
    returns a structured gRPC error — which is enough to catch ingress
    misconfiguration (HTTPProxy timeout/h2c) or auth misconfiguration.
    """
    from nvidia.omniverse.live.v1alpha import live_service_pb2

    session_id = None
    try:
        session_id = create_session(
            live_edit_stub, grpc_metadata, base_path, session_name
        )

        req = live_service_pb2.SubscribeReadLayerRequest(
            session_id=session_id,
            layer_name="smoke-tests-nonexistent-layer",
            object_id=0,
            sequence_num=0,
            client_id=client_id,
        )

        with pytest.raises(grpc.RpcError) as exc_info:
            stream = timeout_grpc_call(
                live_edit_stub.SubscribeReadLayer, req, metadata=grpc_metadata
            )
            # Consume — server-streaming errors surface on iteration.
            for _ in stream:
                pass

        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND, (
            f"Expected NOT_FOUND for missing layer; got {exc_info.value.code()}: "
            f"{exc_info.value.details()}"
        )
        logger.info("SubscribeReadLayer surfaced NOT_FOUND as expected")
    finally:
        if session_id:
            try:
                delete_session(live_edit_stub, grpc_metadata, session_id)
            except Exception:
                pass
