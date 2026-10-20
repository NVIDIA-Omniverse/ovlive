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
"""Build an extracted release decoder with GNU ld and exercise its wire fixtures.

Linux-only, no Docker required. Leaves its isolated archive/build under /tmp
for inspection. Fails on link errors, missing shared libraries, dynamic C++
runtime linkage, decoder crashes or invalid fixture decoding.
"""

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib


def main():
    if sys.platform != "linux":
        raise SystemExit(
            "Run this GNU-linker regression on Linux (or use the Docker build)."
        )
    root = Path(__file__).resolve().parents[1]
    archive = Path(tempfile.mkdtemp(prefix="conformance-native-"))
    print(f"Native regression archive: {archive}", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/stage-conformance-archive.py"),
            str(archive),
        ],
        check=True,
    )
    env = dict(os.environ)
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    env.pop("CARGO_BUILD_TARGET", None)
    env["RUSTFLAGS"] = "-C linker=cc -C link-arg=-fuse-ld=bfd"
    env["CARGO_TARGET_DIR"] = str(archive / "target")
    built = subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--release",
            "-p",
            "oodf-ffi",
            "--example",
            "decode_ompf",
            "-j4",
            "--message-format=json",
        ],
        cwd=archive / "native-codec",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    decoder = next(
        row["executable"]
        for line in built.stdout.splitlines()
        if (row := json.loads(line)).get("executable")
    )
    linked = subprocess.check_output(["ldd", decoder], text=True)
    if "libstdc++" in linked or "not found" in linked:
        raise RuntimeError(f"Invalid decoder runtime dependencies:\n{linked}")
    fixtures = json.loads(
        (
            archive
            / "conformance_tests/src/live_edit_conformance/fixtures/synthetic.json"
        ).read_text()
    )
    for name, encoded in fixtures.items():
        payload = zlib.decompress(base64.b64decode(encoded, validate=True))
        result = subprocess.run(
            [decoder, "--json", "--validate-externals"],
            input=base64.b64encode(payload),
            capture_output=True,
            timeout=15,
            check=True,
        )
        signature = json.loads(result.stdout)
        if signature.get("sourceFormat") != "usda":
            raise RuntimeError(f"Incorrect decoded fixture: {name}")
    print(
        f"PASS: GNU release link, runtime dependencies, {len(fixtures)} decoded fixtures"
    )
    # The service Docker context excludes every example. Cargo must still
    # accept build-script output when the package has no example targets.
    examples = archive / "native-codec/oodf-ffi/examples"
    excluded = archive / "excluded-examples"
    examples.rename(excluded)
    try:
        subprocess.run(
            [
                "cargo",
                "build",
                "--locked",
                "--release",
                "-p",
                "oodf-ffi",
                "--lib",
                "-j4",
            ],
            cwd=archive / "native-codec",
            env=env,
            check=True,
        )
    finally:
        excluded.rename(examples)
    print("PASS: library-only service build context (examples excluded)")


if __name__ == "__main__":
    main()
