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
"""Explicit, validated configuration shared by source and installed runs."""

import math
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import grpc


def positive(name, default):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def target(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Endpoint must be http[s]://host:port without credentials or path")
    # Accessing port also validates malformed/out-of-range port numbers.
    port = parsed.port or (443 if parsed.scheme == "https" else 50001)
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"{host}:{port}", parsed.scheme == "https"


@dataclass(frozen=True)
class Settings:
    endpoint: str
    base_path: str
    token: str = field(repr=False)
    token_file: str = field(repr=False)
    ca_file: str
    timeout: float
    ready_timeout: float
    decoder: str
    reports: Path
    peer_endpoint: str = ""
    denied_path: str = ""

    @classmethod
    def from_env(cls):
        env = os.environ
        endpoint = env.get("CONFORMANCE_GRPC_ENDPOINT", "http://localhost:50001")
        target(endpoint)
        token = env.get("CONFORMANCE_AUTH_TOKEN", "").strip()
        token_file = env.get("CONFORMANCE_AUTH_TOKEN_FILE", "").strip()
        if bool(token) == bool(token_file):
            raise ValueError(
                "Set exactly one of CONFORMANCE_AUTH_TOKEN or CONFORMANCE_AUTH_TOKEN_FILE"
            )
        base = env.get("CONFORMANCE_BASE_PATH", "").strip()
        if not base:
            raise ValueError("CONFORMANCE_BASE_PATH must name an existing authorized base layer")
        peer = env.get("CONFORMANCE_PEER_ENDPOINT", "").strip()
        if peer:
            target(peer)
        settings = cls(
            endpoint,
            base,
            token,
            token_file,
            env.get("CONFORMANCE_CA_FILE", ""),
            positive("CONFORMANCE_TIMEOUT", "15"),
            positive("CONFORMANCE_READY_TIMEOUT", "120"),
            env.get("CONFORMANCE_DECODER", "decode_ompf"),
            Path(env.get("CONFORMANCE_REPORT_DIR", "reports")).resolve(),
            peer,
            env.get("CONFORMANCE_DENIED_PATH", ""),
        )
        return settings

    def metadata(self):
        # Read rotating secrets at each RPC, not just once at process startup.
        token = Path(self.token_file).read_text().strip() if self.token_file else self.token
        if token.startswith("{"):
            token = json.loads(token).get("access_token")
        if not isinstance(token, str):
            raise ValueError("Token JSON must contain a string access_token")
        if not token or any(ord(c) < 33 or ord(c) > 126 for c in token):
            raise ValueError("Bearer token must be nonempty printable ASCII without whitespace")
        return [("authorization", f"Bearer {token}")]

    def channel(self, endpoint=None):
        address, secure = target(endpoint or self.endpoint)
        options = [("grpc.max_receive_message_length", 128 * 1024 * 1024)]
        if secure:
            roots = Path(self.ca_file).read_bytes() if self.ca_file else None
            return grpc.secure_channel(address, grpc.ssl_channel_credentials(roots), options)
        if self.ca_file:
            raise ValueError("CONFORMANCE_CA_FILE requires an https endpoint")
        return grpc.insecure_channel(address, options)
