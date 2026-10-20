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
"""Run installed tests, preserving failures and machine-readable evidence."""

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
import pytest

from .config import Settings
from .protocol import decode, fixture


def wait_ready(settings, endpoint):
    deadline = time.monotonic() + settings.ready_timeout
    with settings.channel(endpoint) as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"LiveService did not become SERVING at {endpoint}")
            try:
                response = stub.Check(
                    health_pb2.HealthCheckRequest(
                        service="nvidia.omniverse.live.v1alpha.LiveService"
                    ),
                    timeout=min(remaining, settings.timeout),
                )
                if response.status == health_pb2.HealthCheckResponse.SERVING:
                    return
                # Alternative implementations need not implement gRPC health.
            except grpc.RpcError as exc:
                if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                    from nvidia.omniverse.live.v1alpha import live_service_pb2 as pb
                    from nvidia.omniverse.live.v1alpha import live_service_pb2_grpc as rpc

                    rpc.LiveServiceStub(channel).Ping(
                        pb.PingRequest(),
                        metadata=settings.metadata(),
                        timeout=min(remaining, settings.timeout),
                    )
                    return
                if exc.code() not in {
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                }:
                    raise RuntimeError(f"Readiness failed with {exc.code().name}") from None
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))


class Coverage:
    def __init__(self, profiles):
        self.profiles = profiles
        self.skipped = 0
        self.executed = 0
        self.unexpected_deselections = 0
        self.expected_deselections = set()

    def pytest_configure(self, config):
        for marker in ("deployment", "permissions", "multi_replica"):
            config.addinivalue_line("markers", f"{marker}: explicitly selected conformance profile")

    def pytest_collection_modifyitems(self, config, items):
        selected, deselected = [], []
        for item in items:
            required = {
                m
                for m in ("deployment", "permissions", "multi_replica")
                if item.get_closest_marker(m)
            }
            (selected if required <= self.profiles else deselected).append(item)
        items[:] = selected
        self.expected_deselections.update(item.nodeid for item in deselected)
        config.hook.pytest_deselected(items=deselected)

    def pytest_deselected(self, items):
        self.unexpected_deselections += sum(
            item.nodeid not in self.expected_deselections for item in items
        )

    def pytest_runtest_logreport(self, report):
        self.skipped += report.skipped
        if report.when == "call":
            self.executed += 1


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        choices=["deployment", "permissions", "multi_replica"],
    )
    args = parser.parse_args()
    reports = Path(os.environ.get("CONFORMANCE_REPORT_DIR", "reports")).resolve()
    reports.mkdir(parents=True, exist_ok=True)
    try:
        if sys.flags.optimize:
            raise ValueError(
                "Conformance requires Python assertions; do not use -O or PYTHONOPTIMIZE"
            )
        settings = Settings.from_env()
        token_deadline = time.monotonic() + settings.ready_timeout
        while True:
            try:
                settings.metadata()
                break
            except FileNotFoundError:
                if time.monotonic() >= token_deadline:
                    raise ValueError(
                        "Token file did not appear within the readiness deadline"
                    ) from None
                time.sleep(0.25)
        if "permissions" in args.profile and not settings.denied_path:
            raise ValueError("permissions profile requires CONFORMANCE_DENIED_PATH")
        if "multi_replica" in args.profile and (
            not settings.peer_endpoint or settings.peer_endpoint == settings.endpoint
        ):
            raise ValueError("multi_replica profile requires a distinct CONFORMANCE_PEER_ENDPOINT")
        decode(settings, fixture("initial"))  # Required decoder and package-data preflight.
        for endpoint in {settings.endpoint, settings.peer_endpoint} - {""}:
            wait_ready(settings, endpoint)
    except Exception as exc:
        # Configuration errors are visible in JUnit too. Never print token-bearing RPC details.
        message = str(exc) if not isinstance(exc, grpc.RpcError) else exc.code().name
        suite = ET.Element("testsuite", name="conformance-preflight", tests="1", errors="1")
        case = ET.SubElement(suite, "testcase", name="preflight")
        ET.SubElement(case, "error", message=message)
        ET.ElementTree(suite).write(reports / "conformance-results.xml", encoding="utf-8")
        print(f"Conformance preflight failed: {message}", file=sys.stderr)
        return 2
    coverage = Coverage(set(args.profile))
    # Certification is independent of developer pytest settings/plugins.
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = int(
        pytest.main(
            [
                str(Path(__file__).parent / "tests"),
                "--import-mode=importlib",
                f"--rootdir={Path(__file__).parent / 'tests'}",
                f"--confcutdir={Path(__file__).parent / 'tests'}",
                "-p",
                "no:cacheprovider",
                "-c",
                os.devnull,
                "-p",
                "pytest_timeout",
                "-v",
                "--tb=short",
                "-o",
                "addopts=",
                "--timeout=120",
                "--timeout-method=thread",
                f"--junitxml={reports / 'conformance-results.xml'}",
            ],
            plugins=[coverage],
        )
    )
    # A skipped required case must never certify an implementation.
    if coverage.skipped or not coverage.executed or coverage.unexpected_deselections:
        result = result or 1
    (reports / "summary.json").write_text(
        json.dumps(
            {
                "profiles": ["core", *args.profile],
                "executed": coverage.executed,
                "skipped": coverage.skipped,
                "unexpected_deselections": coverage.unexpected_deselections,
                "exit_code": result,
                "build": json.loads((Path(__file__).parent / "build.json").read_text()),
            },
            indent=2,
        )
        + "\n"
    )
    return result


def record_failure(name, message):
    reports = Path(os.environ.get("CONFORMANCE_REPORT_DIR", "reports"))
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "conformance-results.xml"
    root = ET.Element("testsuites")
    if path.is_file():
        existing = ET.parse(path).getroot()
        if existing.tag == "testsuites":
            root.extend(existing)
        else:
            root.append(existing)
    suite = ET.SubElement(root, "testsuite", name=name, tests="1", errors="1")
    case = ET.SubElement(suite, "testcase", name=name)
    ET.SubElement(case, "error", message=message)
    ET.ElementTree(root).write(path, encoding="utf-8")
    summary = reports / "summary.json"
    result = json.loads(summary.read_text()) if summary.is_file() else {}
    result.update(exit_code=2, failure=name)
    summary.write_text(json.dumps(result) + "\n")


def emit_reports():
    if os.environ.get("CONFORMANCE_REPORT_STDOUT") == "true":
        reports = Path(os.environ.get("CONFORMANCE_REPORT_DIR", "reports"))
        for name in ("conformance-results.xml", "summary.json"):
            path = reports / name
            if path.is_file():
                print(
                    f"CONFORMANCE_REPORT {name} {base64.b64encode(path.read_bytes()).decode()}",
                    flush=True,
                )


def main():
    try:
        return run()
    finally:
        emit_reports()


if __name__ == "__main__":
    sys.exit(main())
