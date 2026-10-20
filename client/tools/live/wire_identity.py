#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build identity for OMPF envelope/USD value components and their explicit SDKs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.live.sdk_tree_identity import tree_hash as sdk_tree_hash


USD_TREES = ("usd_include", "usd_lib", "python_include", "python_lib")


def inputs(root, boundary="wire_envelope"):
    paths = {"CMakeLists.txt", "cmake/LiveWire.cmake",
             "tools/live/wire_identity.py", "tools/live/sdk_tree_identity.py", "tools/live/test.py", "tools/live/test_support.py",
             "tools/live/process_guard.py", "tools/live/sanitizer.py", "source/live/support/SHA256.h",
             "source/live/support/SHA256.cpp"}
    for directory in ("source/live/wire", "tests/native/wire"):
        paths.update(p.relative_to(root).as_posix() for p in (root / directory).rglob("*") if p.is_file())
    paths.add("cmake/LiveTestSupport.cmake")
    paths.update(p.relative_to(root).as_posix() for p in (root / "tests/native/support").iterdir() if p.is_file())
    if boundary in {"usd_values", "wire_scene", "scene_journal", "scene_transport"}:
        paths.update({"cmake/LiveUsdWire.cmake", "cmake/LiveSceneCore.cmake", "cmake/LiveUsdSdk.cmake", "cmake/LiveSanitizer.cmake"})
        for directory in ("tools/live", "tests/harness", "source/live/support", "include/omni/usd/live"):
            paths.update(p.relative_to(root).as_posix() for p in (root / directory).rglob("*")
                         if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    if boundary in {"wire_scene", "scene_journal", "scene_transport"}:
        paths.add("cmake/LiveWireScene.cmake")
        for directory in ("source/live/scene", "source/live/support", "tests/native/scene"):
            paths.update(p.relative_to(root).as_posix() for p in (root / directory).rglob("*") if p.is_file())
    if boundary in {"scene_journal", "scene_transport"}:
        paths.update({"source/live/runtime/LiveLayerProgress.h", "source/live/runtime/LiveLayerProgress.cpp",
                      "source/live/runtime/LiveUri.h", "source/live/runtime/LiveUri.cpp", "source/live/runtime/LiveUriApi.cpp",
                      "include/omni/usd/live/Live.h"})
    if boundary == "scene_transport":
        paths.update({"cmake/LiveSceneTransport.cmake", "cmake/LiveGrpcDependencies.cmake",
                      "cmake/LiveBuildProvenance.h.in"})
        for directory in ("source/live/runtime", "source/live/resolver", "source/live/protos", "tests/native/runtime", "tests/reference_service"):
            paths.update(p.relative_to(root).as_posix() for p in (root / directory).rglob("*")
                         if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    return sorted(paths)


def tree_hash(root):
    if not root.is_dir():
        raise ValueError(f"missing SDK tree: {root}")
    return sdk_tree_hash(root)


def sdk_identity(root, include):
    provenance = json.loads((root / "source/live/wire/schema/provenance.json").read_text())["flatbuffers"]
    headers = provenance["headers"]
    if set(headers) != {"flatbuffers.h", "base.h", "stl_emulation.h"}:
        raise ValueError("incomplete FlatBuffers header provenance")
    for name, expected in headers.items():
        if hashlib.sha256((include / "flatbuffers" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"FlatBuffers {name} differs from provenance pin")
    return {"package": provenance["package"], "version": provenance["version"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "artifact", "flatbuffers"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--boundary", choices=("wire_envelope", "usd_values", "wire_scene", "scene_journal", "scene_transport"), default="wire_envelope")
    parser.add_argument("--dependency", type=Path, action="append", default=[])
    for field in ("usd", "python"):
        parser.add_argument("--" + field, type=Path)
    parser.add_argument("--compiler-id", required=True)
    parser.add_argument("--compiler-version", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--sanitizers", choices=("0", "1"), required=True)
    parser.add_argument("--sanitizer", choices=("none", "address", "thread"))
    args = parser.parse_args()
    # --sanitizers 1 without a kind is the address build of the component workflow.
    kind = args.sanitizer or ("address" if args.sanitizers == "1" else "none")
    if (kind != "none") != (args.sanitizers == "1"):
        parser.error("--sanitizer and --sanitizers disagree")
    root = args.root.resolve()
    include = (args.flatbuffers / "include").resolve()
    trees = [("flatbuffers_include", include)]
    if args.boundary in {"usd_values", "wire_scene", "scene_journal", "scene_transport"}:
        for field in ("usd", "python"):
            if getattr(args, field) is None:
                parser.error(f"--{field} is required for {args.boundary}")
        trees.extend((role, (getattr(args, role.split("_")[0]) / role.split("_")[1]).resolve())
                     for role in USD_TREES)
    if args.boundary == "scene_transport" and not args.dependency:
        parser.error("scene_transport requires explicit gRPC dependency files")
    manifest = {
        "schema_version": 1, "boundary": args.boundary, "configuration": args.configuration,
        "compiler_id": args.compiler_id, "compiler_version": args.compiler_version,
        "sanitizers": args.sanitizers == "1", "sanitizer": kind, "flatbuffers": sdk_identity(root, include),
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "inputs": [{"path": p, "sha256": hashlib.sha256((root / p).read_bytes()).hexdigest()}
                   for p in inputs(root, args.boundary)],
        "dependency_trees": [{"role": role, "path": str(path), "sha256": tree_hash(path)}
                             for role, path in trees],
    }
    if args.dependency:
        manifest["dependencies"] = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                    for path in sorted({p.resolve() for p in args.dependency})]
    destination = Path(str(args.artifact) + ".manifest.json")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
