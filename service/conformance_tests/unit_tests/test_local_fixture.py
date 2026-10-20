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
"""Local/CI identity needs no external backend and never becomes a default."""

import base64
import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml

from live_edit_conformance.config import Settings

ROOT = Path(__file__).resolve().parents[2]
OVERRIDE = "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS"
TOKEN = ROOT / "conformance_tests/local-test-token.jwt"


def test_local_token_has_explicit_unsigned_identity():
    header, claims, signature = TOKEN.read_text().strip().split(".")
    def decode(value):
        return json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))

    assert decode(header) == {"alg": "none", "typ": "JWT"}
    assert signature == ""
    assert decode(claims) == {
        "sub": "conformance-alice", "name": "Alice", "preferred_username": "alice",
    }


def assert_isolated_service(env):
    assert env["LIVE_EDIT_AUTHN_ENABLED"] == "true"
    assert env["LIVE_EDIT_AUTHZ_ENABLED"] == "false"
    assert env["LIVE_EDIT_USERINFO_SERVICE_ENABLED"] == "false"
    assert env[OVERRIDE] == "true"
    assert "LIVE_EDIT_USERINFO_ENDPOINT" not in env
    assert "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT" not in env
    assert env["LIVE_EDIT_VALKEY_POOL_SIZE"] == "8"
    assert env["LIVE_EDIT_SUBSCRIBER_POOL_SIZE"] == "4"


def test_compose_has_no_external_identity_dependency():
    services = yaml.safe_load((ROOT / "conformance_tests/docker-compose.yml").read_text())["services"]
    assert set(services) == {"valkey", "service", "conformance"}
    assert_isolated_service(services["service"]["environment"])
    assert set(services["service"]["depends_on"]) == {"valkey"}
    runner = services["conformance"]["environment"]
    assert runner["CONFORMANCE_AUTH_TOKEN_FILE"] == "/opt/local-test-token.jwt"
    assert "CONFORMANCE_AUTH_TOKEN" not in runner
    assert OVERRIDE not in runner
    # The dangerous mode is not exposed on host ports or enabled by Helm.
    assert all("ports" not in service for service in services.values())
    for path in (ROOT / "helm").rglob("*.yaml"):
        assert OVERRIDE not in path.read_text(), path


@pytest.mark.parametrize("replicas", [1, 2])
def test_local_launcher_starts_only_service_and_runner(tmp_path, monkeypatch, replicas):
    spec = importlib.util.spec_from_file_location(
        "local_conformance_launcher", ROOT / "scripts/run-conformance-local.py"
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    calls = []

    class Process:
        def __init__(self, args, *, env, **kwargs):
            self.args, self.env = args, dict(env)
            self.returncode = 0 if "live_edit_conformance.runner" in args else None
            calls.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr("sys.argv", [
        "run-conformance-local.py", "--replicas", str(replicas), "--report-dir", str(tmp_path),
    ])
    monkeypatch.setenv("LIVE_EDIT_USERINFO_ENDPOINT", "https://production.invalid")
    monkeypatch.setenv("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT", "https://production.invalid")
    monkeypatch.setenv("CONFORMANCE_AUTH_TOKEN", "must-not-be-used")
    monkeypatch.setenv("CONFORMANCE_AUTH_TOKEN_FILE", "/must-not-be-read")
    monkeypatch.setenv(OVERRIDE, "false")
    assert launcher.main() == 0
    assert len(calls) == replicas + 1
    for process in calls[:-1]:
        assert len(process.args) == 1 and process.args[0].endswith("/live-edit")
        assert_isolated_service(process.env)
        assert process.returncode == 0  # Owned service processes were stopped.
    runner = calls[-1]
    assert runner.env["CONFORMANCE_AUTH_TOKEN_FILE"] == str(TOKEN)
    assert "CONFORMANCE_AUTH_TOKEN" not in runner.env
    assert not (tmp_path / "identity.log").exists()
    assert ("multi_replica" in runner.args) == (replicas == 2)


def test_public_runner_requires_credentials_and_can_read_local_fixture(monkeypatch):
    for key in list(os.environ):
        if key.startswith("CONFORMANCE_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("CONFORMANCE_BASE_PATH", "https://example.test/conformance/base.usda")
    with pytest.raises(ValueError, match="exactly one"):
        Settings.from_env()
    monkeypatch.setenv("CONFORMANCE_AUTH_TOKEN_FILE", str(TOKEN))
    assert Settings.from_env().metadata() == [("authorization", "Bearer " + TOKEN.read_text().strip())]


def test_archive_includes_local_token_not_identity_server(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "stage_conformance_archive", ROOT / "scripts/stage-conformance-archive.py"
    )
    staging = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staging)
    staging.stage(tmp_path)
    assert (tmp_path / "conformance_tests/local-test-token.jwt").read_bytes() == TOKEN.read_bytes()
    assert not (tmp_path / "conformance_tests/identity_fixture.py").exists()


def test_archive_ships_notice_wherever_it_ships_the_license(tmp_path):
    # Apache-2.0 §4(d): NVIDIA's attribution (NOTICE) goes with every copy of
    # the license text, at the archive root and in the native codec.
    spec = importlib.util.spec_from_file_location(
        "stage_conformance_archive", ROOT / "scripts/stage-conformance-archive.py"
    )
    staging = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staging)
    staging.stage(tmp_path)
    for directory in (tmp_path, tmp_path / "native-codec"):
        for name in ("LICENSE.txt", "NOTICE"):
            assert (directory / name).read_bytes() == (ROOT / name).read_bytes(), directory / name
