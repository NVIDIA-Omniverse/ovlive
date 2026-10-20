#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record native test artifact identity and its explicitly listed source inputs."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--compiler-id", required=True)
    parser.add_argument("--compiler-version", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--sanitizers", choices=("0", "1"), required=True)
    parser.add_argument("--sanitizer", choices=("none", "address", "thread"))
    parser.add_argument("--boundary", choices=("uri_component", "runtime_foundation", "unary_transport", "lock_noalloc"), default="uri_component")
    parser.add_argument("--dependency", action="append", default=[])
    parser.add_argument("--input", action="append", required=True)
    args = parser.parse_args()
    # --sanitizers 1 without a kind is the address build of the component workflow.
    kind = args.sanitizer or ("address" if args.sanitizers == "1" else "none")
    if (kind != "none") != (args.sanitizers == "1"):
        parser.error("--sanitizer and --sanitizers disagree")
    root = args.root.resolve()
    inputs = []
    for name in args.input:
        path = (root / name).resolve()
        path.relative_to(root)  # Build inputs cannot escape the checkout.
        inputs.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    dependencies = []
    for name in args.dependency:
        path = Path(name).resolve()
        dependencies.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {"schema_version": 1, "boundary": args.boundary,
              "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
              "compiler_id": args.compiler_id, "compiler_version": args.compiler_version,
              "configuration": args.configuration, "sanitizers": args.sanitizers == "1", "sanitizer": kind, "inputs": inputs,
              "dependencies": dependencies}
    destination = Path(str(args.artifact) + ".manifest.json")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
