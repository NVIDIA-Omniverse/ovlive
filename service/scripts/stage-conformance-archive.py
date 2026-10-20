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
"""Stage an explicit, self-contained source distribution in the API archive."""

from pathlib import Path
import shutil
import sys


def stage(destination):
    root = Path(__file__).resolve().parents[1]
    destination = Path(destination)
    suite = destination / "conformance_tests"
    suite.mkdir(parents=True, exist_ok=True)
    for name in ["pyproject.toml", "uv.lock", "README.md", "local-test-token.jwt"]:
        shutil.copyfile(root / "conformance_tests" / name, suite / name)
    shutil.copytree(
        root / "conformance_tests/src",
        suite / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )
    codec = destination / "native-codec"
    codec.mkdir(exist_ok=True)
    shutil.copyfile(
        root / "conformance_tests/native-workspace.toml", codec / "Cargo.toml"
    )
    shutil.copyfile(root / "conformance_tests/native-Cargo.lock", codec / "Cargo.lock")
    for name in ["LICENSE.txt", "NOTICE"]:
        shutil.copyfile(root / name, codec / name)
    library = codec / "oodf-ffi"
    library.mkdir(exist_ok=True)
    for name in ["Cargo.toml", "build.rs"]:
        shutil.copyfile(root / "oodf-ffi" / name, library / name)
    for directory in ["src", "include", "cpp", "schema"]:
        shutil.copytree(
            root / "oodf-ffi" / directory, library / directory, dirs_exist_ok=True
        )
    (library / "examples").mkdir(exist_ok=True)
    shutil.copyfile(
        root / "oodf-ffi/examples/decode_ompf.rs", library / "examples/decode_ompf.rs"
    )
    for name in ["LICENSE.txt", "NOTICE", "ATTRIBUTIONS.md"]:
        shutil.copyfile(root / name, destination / name)


if __name__ == "__main__":
    stage(sys.argv[1])
