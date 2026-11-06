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

"""Tests for `scripts/run-conformance-local.py`, the launcher that runs the
conformance suite binary against one or two local service replicas.

`subprocess.Popen` is replaced, so nothing is started: each test checks the
processes the launcher would start and the environment each gets. The suite
is the `--suite` binary (default `target/debug/live-edit-conformance`), run
directly with `--profile deployment`, plus `--profile multi_replica` with
`--replicas 2`. The service replicas get the isolated local identity and none
of the caller's `LIVE_EDIT_*` or `CONFORMANCE_*` settings.

Stdlib `unittest`, matching the `supply-chain:python-tests` job. Run from the
repo root with:

    python3 -m unittest tools.tests.test_run_conformance_local
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "run-conformance-local.py"
TOKEN = REPO_ROOT / "conformance_tests" / "local-test-token.jwt"
OVERRIDE = "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS"

#: Ambient settings that must not reach the run: production identity
#: backends, a caller's credentials and a disabled override.
AMBIENT = {
    "LIVE_EDIT_USERINFO_ENDPOINT": "https://production.invalid",
    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT": "https://production.invalid",
    "CONFORMANCE_AUTH_TOKEN": "must-not-be-used",
    "CONFORMANCE_AUTH_TOKEN_FILE": "/must-not-be-read",
    OVERRIDE: "false",
}


def load_launcher():
    spec = importlib.util.spec_from_file_location("local_conformance_launcher", LAUNCHER)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


class LauncherTests(unittest.TestCase):
    def assert_isolated_service(self, env):
        self.assertEqual(env["LIVE_EDIT_AUTHN_ENABLED"], "true")
        self.assertEqual(env["LIVE_EDIT_AUTHZ_ENABLED"], "false")
        self.assertEqual(env["LIVE_EDIT_USERINFO_SERVICE_ENABLED"], "false")
        self.assertEqual(env[OVERRIDE], "true")
        self.assertNotIn("LIVE_EDIT_USERINFO_ENDPOINT", env)
        self.assertNotIn("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT", env)
        self.assertEqual(env["LIVE_EDIT_VALKEY_POOL_SIZE"], "8")
        self.assertEqual(env["LIVE_EDIT_SUBSCRIBER_POOL_SIZE"], "4")

    def run_launcher(self, reports, replicas, suite_args, suite):
        """Runs the launcher's `main` with `Popen` replaced; returns its exit
        code and the stand-in processes in start order."""
        launcher = load_launcher()
        calls = []

        class Process:
            def __init__(self, args, *, env, **kwargs):
                self.args, self.env = args, dict(env)
                # The suite exits at once; the service replicas run until stopped.
                self.returncode = 0 if args[0] == str(suite) else None
                calls.append(self)

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def wait(self, **kwargs):
                return self.returncode

            def kill(self):
                self.returncode = -9

        argv = [
            "run-conformance-local.py", "--replicas", str(replicas),
            "--report-dir", str(reports), *suite_args,
        ]
        with mock.patch.object(launcher.subprocess, "Popen", Process), \
                mock.patch.object(launcher.sys, "argv", argv), \
                mock.patch.dict(os.environ, AMBIENT):
            code = launcher.main()
        return code, calls

    def test_starts_only_the_service_replicas_and_the_suite(self):
        for replicas in (1, 2):
            for explicit_suite in (False, True):
                with self.subTest(replicas=replicas, explicit_suite=explicit_suite), \
                        tempfile.TemporaryDirectory() as tmp:
                    reports = Path(tmp)
                    if explicit_suite:
                        suite = reports / "bin" / "live-edit-conformance"
                        suite_args = ["--suite", str(suite)]
                    else:
                        suite = Path("target/debug/live-edit-conformance").resolve()
                        suite_args = []
                    code, calls = self.run_launcher(reports, replicas, suite_args, suite)
                    self.assertEqual(code, 0)
                    self.assertEqual(len(calls), replicas + 1)
                    for process in calls[:-1]:
                        self.assertEqual(len(process.args), 1)
                        self.assertTrue(process.args[0].endswith("/live-edit"), process.args)
                        self.assert_isolated_service(process.env)
                        # The launcher stopped the replicas it started.
                        self.assertEqual(process.returncode, 0)
                    runner = calls[-1]
                    # The suite binary runs directly with its profiles.
                    profiles = ["--profile", "deployment"] + (
                        ["--profile", "multi_replica"] if replicas == 2 else []
                    )
                    self.assertEqual(runner.args, [str(suite)] + profiles)
                    self.assertEqual(runner.env["CONFORMANCE_AUTH_TOKEN_FILE"], str(TOKEN))
                    self.assertEqual(runner.env["CONFORMANCE_REPORT_DIR"], str(reports.resolve()))
                    self.assertNotIn("CONFORMANCE_AUTH_TOKEN", runner.env)
                    self.assertNotIn("CONFORMANCE_DECODER", runner.env)
                    if replicas == 2:
                        peer_port = calls[1].env["LIVE_EDIT_PORT"]
                        self.assertEqual(
                            runner.env["CONFORMANCE_PEER_ENDPOINT"], f"http://127.0.0.1:{peer_port}"
                        )
                    else:
                        self.assertNotIn("CONFORMANCE_PEER_ENDPOINT", runner.env)
                    self.assertFalse((reports / "identity.log").exists())


if __name__ == "__main__":
    unittest.main()
