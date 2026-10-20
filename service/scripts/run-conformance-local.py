#!/usr/bin/env python3
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
"""Exercise a production binary locally with the same unsigned test JWT as CI.

Uses TEST_VALKEY_URL and creates only suite-owned sessions. No flush/reset.
"""

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="target/debug/live-edit")
    parser.add_argument("--decoder", default="target/debug/examples/decode_ompf")
    parser.add_argument("--report-dir")
    parser.add_argument("--replicas", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reports = Path(
        args.report_dir or tempfile.mkdtemp(prefix="live-edit-conformance-")
    ).resolve()
    reports.mkdir(parents=True, exist_ok=True)
    grpc_port, metrics_port = port(), port()
    # Prevent ambient production settings/credentials from redirecting the fixture run.
    env = {
        key: val
        for key, val in os.environ.items()
        if not key.startswith(("LIVE_EDIT_", "CONFORMANCE_"))
    }
    env.update(
        {
            "LIVE_EDIT_VALKEY_URL": os.environ.get(
                "TEST_VALKEY_URL", "redis://localhost:6379"
            ),
            "LIVE_EDIT_PORT": str(grpc_port),
            "LIVE_EDIT_METRICS_PORT": str(metrics_port),
            "LIVE_EDIT_AUTHN_ENABLED": "true",
            "LIVE_EDIT_AUTHZ_ENABLED": "false",
            "LIVE_EDIT_USERINFO_SERVICE_ENABLED": "false",
            "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS": "true",
            "LIVE_EDIT_CHECKPOINT_ENABLED": "true",
            "LIVE_EDIT_CHECKPOINT_ADAPTER": "localdisk",
            "LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH": str(reports / "checkpoints"),
            "LIVE_EDIT_VALKEY_POOL_SIZE": "8",
            "LIVE_EDIT_SUBSCRIBER_POOL_SIZE": "4",
            # A short client keepalive interval, so the keepalive case's
            # silent join ends within seconds (three intervals) rather than
            # the default's 45 s.
            "LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS": "1000",
            "CONFORMANCE_GRPC_ENDPOINT": f"http://127.0.0.1:{grpc_port}",
            "CONFORMANCE_AUTH_TOKEN_FILE": str(root / "conformance_tests/local-test-token.jwt"),
            "CONFORMANCE_BASE_PATH": "https://example.test/conformance/base.usda",
            "CONFORMANCE_DECODER": str(Path(args.decoder).resolve()),
            "CONFORMANCE_REPORT_DIR": str(reports),
        }
    )
    processes = []
    try:
        with (reports / "service.log").open("w") as service_log:
            processes.append(
                subprocess.Popen(
                    [str(Path(args.binary).resolve())],
                    env=env,
                    stdout=service_log,
                    stderr=subprocess.STDOUT,
                )
            )
            if args.replicas == 2:
                peer_port = port()
                peer_env = dict(
                    env,
                    LIVE_EDIT_PORT=str(peer_port),
                    LIVE_EDIT_METRICS_PORT=str(port()),
                    LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH=str(
                        reports / "peer-checkpoints"
                    ),
                )
                processes.append(
                    subprocess.Popen(
                        [str(Path(args.binary).resolve())],
                        env=peer_env,
                        stdout=service_log,
                        stderr=subprocess.STDOUT,
                    )
                )
                env["CONFORMANCE_PEER_ENDPOINT"] = f"http://127.0.0.1:{peer_port}"
            runner = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "live_edit_conformance.runner",
                    "--profile",
                    "deployment",
                ]
                + (["--profile", "multi_replica"] if args.replicas == 2 else []),
                env=env,
            )
            processes.append(runner)
            while runner.poll() is None:
                if any(process.poll() is not None for process in processes[:-1]):
                    runner.terminate()
                    runner.wait(timeout=10)
                    print((reports / "service.log").read_text(), file=sys.stderr)
                    return 2
                time.sleep(0.1)
        print(f"Conformance reports: {reports}")
        return runner.returncode
    finally:
        for process in reversed(processes):
            if process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    sys.exit(main())
