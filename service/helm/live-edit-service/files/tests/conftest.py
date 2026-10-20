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

import logging
import os
import random
import string
import uuid

import pytest
from utils import (
    CONFLICTING_AUTH_MESSAGE,
    create_grpc_channel_from_url,
    delete_file_rest,
    get_access_token,
    has_access_token_file,
    has_client_credentials,
    write_file_rest,
)

logging.basicConfig(
    level=logging.INFO, format="%(filename)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""

    pass


def _get_grpc_endpoint() -> str:
    """Get Live Edit Service gRPC endpoint from environment."""
    endpoint = os.getenv("SMOKE_TESTS_GRPC_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        raise ConfigurationError(
            "Live Edit gRPC endpoint not configured. "
            "Set SMOKE_TESTS_GRPC_ENDPOINT (e.g. http://localhost:50001)."
        )
    logger.info("Using Live Edit gRPC endpoint: %s", endpoint)
    return endpoint


def _get_resource_base() -> str:
    """Get the base storage path the smoke principal has write access on.

    The smoke fixture appends a per-session random suffix and PUTs a small
    fixture file under this prefix, so `CreateSession` has a real
    storage-api file to authorize against.
    """
    base = os.getenv("SMOKE_TESTS_RESOURCE_BASE", "").strip()
    if not base:
        raise ConfigurationError(
            "Smoke test resource base not configured. Set "
            "SMOKE_TESTS_RESOURCE_BASE to a base storage path the smoke "
            "principal has write permission on (e.g. "
            "https://storage.example.com/mybucket/)."
        )
    if not base.endswith("/"):
        base += "/"
    logger.info("Using smoke-test resource base: %s", base)
    return base


def _get_storage_rest_endpoint() -> str:
    """Get the storage-api REST endpoint from environment.

    Reused from the live-edit service's own `authorization.storageServiceEndpoint`
    chart value — the smoke pod talks to the same storage-api instance the
    service does, so the smoke principal's existing storage access carries
    over to the fixture-file lifecycle.
    """
    endpoint = os.getenv("SMOKE_TESTS_STORAGE_REST_ENDPOINT", "").strip()
    if not endpoint:
        raise ConfigurationError(
            "Storage-api REST endpoint not configured. Set "
            "SMOKE_TESTS_STORAGE_REST_ENDPOINT (typically the same value as "
            "the chart's `authorization.storageServiceEndpoint`)."
        )
    logger.info("Using storage-api REST endpoint: %s", endpoint)
    return endpoint


def pytest_configure(config):
    """Validate environment variables before running tests."""
    logger.info("\n%s", "=" * 70)
    logger.info("PREFLIGHT CHECKS")
    logger.info("%s\n", "=" * 70)

    if has_access_token_file() and has_client_credentials():
        logger.error(CONFLICTING_AUTH_MESSAGE)
        pytest.exit("Conflicting auth configuration", returncode=1)

    _get_grpc_endpoint()
    _get_resource_base()
    _get_storage_rest_endpoint()

    logger.info("\n%s", "=" * 70)
    logger.info("All preflight checks passed!")
    logger.info("%s\n", "=" * 70)


@pytest.fixture(scope="session")
def grpc_endpoint() -> str:
    """Session-scoped fixture for the Live Edit gRPC base URL."""
    return _get_grpc_endpoint()


@pytest.fixture(scope="session")
def _bearer_token():
    """Resolve the bearer token once per smoke run.

    Session-scoped: the token is captured at suite start and reused
    throughout. This is safe for a short smoke run; if the suite ever
    outlives the token TTL it would break even though the sidecar keeps
    refreshing the token file.
    """
    return get_access_token()


@pytest.fixture(scope="session")
def grpc_metadata(_bearer_token):
    """Create gRPC metadata with optional Bearer token."""
    return (
        [("authorization", f"Bearer {_bearer_token}")] if _bearer_token else []
    )


@pytest.fixture(scope="session")
def http_headers(_bearer_token):
    """HTTP headers for storage-api REST calls (PUT / DELETE fixture file)."""
    headers = {"Content-Type": "application/octet-stream"}
    if _bearer_token:
        headers["Authorization"] = f"Bearer {_bearer_token}"
    return headers


@pytest.fixture(scope="session")
def storage_rest_endpoint() -> str:
    """Session-scoped storage-api REST endpoint."""
    return _get_storage_rest_endpoint()


@pytest.fixture(scope="session")
def grpc_channel(grpc_endpoint):
    """Create and manage gRPC channel for Live Edit Service."""
    channel = create_grpc_channel_from_url(grpc_endpoint)
    yield channel
    channel.close()


@pytest.fixture(scope="session")
def live_edit_stub(grpc_channel):
    """Create LiveService gRPC stub."""
    from nvidia.omniverse.live.v1alpha import live_service_pb2_grpc

    return live_service_pb2_grpc.LiveServiceStub(grpc_channel)


@pytest.fixture(scope="session")
def base_path(http_headers, storage_rest_endpoint) -> str:
    """Provision (and tear down) a real storage-api file at the smoke path.

    `CreateSession` authorizes against storage-api by stat-ing the supplied
    path (`HEAD /v1alpha/fileobject/by-address/<encoded>`); a path that does
    not exist returns 404 → `NOT_FOUND`. So before any session-using test
    runs we PUT a tiny fixture file at `<SMOKE_TESTS_RESOURCE_BASE>/live-edit-
    <random>.live` via storage-api REST. All tests in the suite share this
    one path (each test uses a distinct random `session_name`, so the Valkey
    sessions-by-path-and-name uniqueness index is not stressed).

    Teardown runs unconditionally — even if every test fails — and any
    cleanup error is logged but does not propagate, so a flaky storage-api
    DELETE at the end of an otherwise-passing run cannot fail the suite.
    The random suffix is per-session, so an aborted run leaving an orphan
    will not collide with subsequent runs.
    """
    base = _get_resource_base()
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    path = f"{base}live-edit-{suffix}.live"

    logger.info("Provisioning smoke-test fixture file: %s", path)
    write_file_rest(
        path,
        b"live-edit smoke test fixture\n",
        http_headers,
        storage_rest_endpoint,
    )
    try:
        yield path
    finally:
        try:
            delete_file_rest(path, http_headers, storage_rest_endpoint)
            logger.info("Smoke-test fixture file cleaned up")
        except Exception as e:
            logger.warning(
                "Failed to clean up smoke-test fixture file %s: %s", path, e
            )


@pytest.fixture()
def session_name() -> str:
    """Generate a unique session name for testing."""
    return f"smoke-session-{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def client_id() -> str:
    """Generate a unique client id for self-echo suppression."""
    return f"smoke-client-{uuid.uuid4().hex}"
