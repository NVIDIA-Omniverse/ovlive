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
"""Regenerate committed bindings, wire fixtures and their provenance manifest."""

import base64
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zlib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "conformance_tests/src/live_edit_conformance"


def main():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{ROOT / 'proto'}",
            f"--python_out={ROOT / 'conformance_tests/src'}",
            f"--grpc_python_out={ROOT / 'conformance_tests/src'}",
            str(ROOT / "proto/nvidia/omniverse/live/v1alpha/live_service.proto"),
        ],
        check=True,
    )
    stubs = ROOT / "conformance_tests/src/nvidia/omniverse/live/v1alpha"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/license_headers.py"),
            "fix",
            str(stubs / "live_service_pb2.py"),
            str(stubs / "live_service_pb2_grpc.py"),
        ],
        check=True,
    )
    built = subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "-p",
            "oodf-ffi",
            "--example",
            "conformance_fixture",
            "--message-format=json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    executable = next(
        row["executable"]
        for line in built.stdout.splitlines()
        if (row := json.loads(line)).get("executable")
    )
    fixtures = {}
    for name in [
        "initial",
        "update",
        "time_sample",
        "update_time",
        "external",
        "large",
        "structure",
    ]:
        raw = subprocess.check_output([executable, name])
        parts = [raw] + ([b"C" * 8192] if name == "external" else [])
        offset = 8 + len(parts) * 8
        offsets = []
        for part in parts:
            offsets.append(offset)
            offset += len(part)
        ompf = (
            b"OMPF"
            + struct.pack("<I", len(parts))
            + struct.pack(f"<{len(parts)}Q", *offsets)
            + b"".join(parts)
        )
        fixtures[name] = base64.b64encode(zlib.compress(ompf, 9)).decode()
    directory = PACKAGE / "fixtures"
    directory.mkdir(exist_ok=True)
    (directory / "synthetic.json").write_text(json.dumps(fixtures, indent=2) + "\n")
    shutil.copyfile(
        ROOT / "tools/fixtures/gze_inbound_deltas.json", directory / "captured.json"
    )
    shutil.copyfile(
        ROOT / "tools/fixtures/gze_cpp_catchup_oracle.json",
        directory / "captured-golden.json",
    )
    manifest = {"contract": "nvidia.omniverse.live.v1alpha.LiveService", "sha256": {}}
    for name, path in {
        "proto": ROOT / "proto/nvidia/omniverse/live/v1alpha/live_service.proto",
        "oodf_schema": ROOT / "oodf-ffi/schema/OmniObjectDelta.fbs",
        "synthetic": directory / "synthetic.json",
        "captured": directory / "captured.json",
        "captured_golden": directory / "captured-golden.json",
    }.items():
        manifest["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (PACKAGE / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
