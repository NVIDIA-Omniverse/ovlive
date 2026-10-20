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

"""Common test helpers for Live Edit Service gRPC smoke tests."""

import logging
import time

from utils import timeout_grpc_call

logger = logging.getLogger(__name__)


def create_session(stub, metadata, path, name):
    """Create a session and return its session_id."""
    from nvidia.omniverse.live.v1alpha import live_service_pb2

    req = live_service_pb2.CreateSessionRequest(path=path, name=name)
    resp = timeout_grpc_call(stub.CreateSession, req, metadata=metadata)
    assert resp.session_id, "CreateSession returned empty session_id"
    return resp.session_id


def delete_session(stub, metadata, session_id, *, retry_budget_seconds=60):
    """Delete a session. Retry up to retry_budget_seconds if it fails."""
    from nvidia.omniverse.live.v1alpha import live_service_pb2

    req = live_service_pb2.DeleteSessionRequest(session_id=session_id)
    last_error = None
    start = time.time()
    while time.time() - start < retry_budget_seconds:
        try:
            timeout_grpc_call(stub.DeleteSession, req, metadata=metadata)
            return
        except Exception as e:
            last_error = e
            time.sleep(2)
    raise AssertionError(
        f"Failed to delete session {session_id} after "
        f"{retry_budget_seconds}s: {last_error}"
    )


def list_sessions(stub, metadata, path):
    """List sessions for a base layer path."""
    from nvidia.omniverse.live.v1alpha import live_service_pb2

    req = live_service_pb2.ListSessionsRequest(path=path)
    return timeout_grpc_call(stub.ListSessions, req, metadata=metadata)
