#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identity of the test-only USD resolver driver, plugin, metadata and SDKs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.live.sdk_tree_identity import tree_hash as sdk_tree_hash

SDK_TREES = {"usd_include", "usd_lib", "python_include", "python_lib"}


def inputs(root):
    paths = {"CMakeLists.txt", "cmake/LiveSceneCore.cmake", "cmake/LiveUsdSdk.cmake", "cmake/LiveSanitizer.cmake", "cmake/LiveResolverProbe.cmake",
             "tools/live/resolver_identity.py", "tools/live/sdk_tree_identity.py", "tools/live/test.py", "include/omni/usd/live/Live.h",
             "source/live/runtime/LiveUri.h", "source/live/runtime/LiveUri.cpp", "source/live/runtime/LiveUriApi.cpp"}
    paths.update(p.relative_to(root).as_posix() for p in (root / "tests/native/resolver").rglob("*") if p.is_file())
    paths.add("cmake/LiveTestSupport.cmake")
    paths.update(p.relative_to(root).as_posix() for p in (root / "tests/native/support").iterdir() if p.is_file())
    return sorted(paths)


def tree_hash(root):
    if not root.is_dir():
        raise ValueError("resolver SDK directory is missing")
    return sdk_tree_hash(root)


def validate_dependencies(manifest):
    dependencies = manifest.get("dependencies", [])
    if len(dependencies) != 2 or {entry.get("role") for entry in dependencies} != {"plugin", "plug_info"}:
        raise ValueError("resolver manifest is missing plugin/metadata identity")
    trees = manifest.get("dependency_trees", [])
    if len(trees) != len(SDK_TREES) or {entry.get("role") for entry in trees} != SDK_TREES:
        raise ValueError("resolver manifest is missing SDK identity")
    for entry in trees:
        if tree_hash(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("native resolver SDK differs; rebuild required")
    paths = {entry["role"]: Path(entry["path"]) for entry in dependencies}
    plugin = json.loads(paths["plug_info"].read_text())["Plugins"]
    if len(plugin) != 1 or Path(plugin[0]["LibraryPath"]).resolve() != paths["plugin"].resolve():
        raise ValueError("resolver plugin metadata points to an unrecorded binary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "artifact", "plugin", "plug-info", "usd", "python"):
        parser.add_argument("--" + field, type=Path, required=True)
    for field in ("compiler-id", "compiler-version", "configuration"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--sanitizers", choices=("0", "1"), required=True)
    parser.add_argument("--sanitizer", choices=("none", "address", "thread"))
    args = parser.parse_args()
    # --sanitizers 1 without a kind is the address build of the component workflow.
    kind = args.sanitizer or ("address" if args.sanitizers == "1" else "none")
    if (kind != "none") != (args.sanitizers == "1"):
        parser.error("--sanitizer and --sanitizers disagree")
    root = args.root.resolve()
    trees = []
    for role in sorted(SDK_TREES):
        sdk, directory = role.split("_")
        path = (getattr(args, sdk) / directory).resolve()
        trees.append({"role": role, "path": str(path), "sha256": tree_hash(path)})
    manifest = {
        "schema_version": 1, "boundary": "resolver_probe", "configuration": args.configuration,
        "compiler_id": args.compiler_id, "compiler_version": args.compiler_version,
        "sanitizers": args.sanitizers == "1", "sanitizer": kind,
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "inputs": [{"path": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
                   for name in inputs(root)],
        "dependencies": [{"role": role, "path": str(path.resolve()),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for role, path in (("plugin", args.plugin), ("plug_info", args.plug_info))],
        "dependency_trees": trees,
    }
    validate_dependencies(manifest)
    destination = Path(str(args.artifact) + ".manifest.json")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
