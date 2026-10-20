#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identity of the production resolver/runtime component and reference driver.

This boundary does not qualify the Sdf file format or complete live product.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.live.resolver_identity import SDK_TREES, tree_hash


def inputs(root):
    paths = {"CMakeLists.txt", "cmake/LiveResolverRuntime.cmake", "cmake/LiveSceneCore.cmake", "cmake/LiveUsdSdk.cmake", "cmake/LiveSanitizer.cmake",
             "cmake/LiveGrpcDependencies.cmake", "cmake/LiveBuildProvenance.h.in",
             "include/omni/usd/live/Live.h", "source/live/support/LiveTsan.h"}
    for directory in ("source/live/runtime", "source/live/protos", "source/live/resolver",
                      "tests/native/resolver_runtime", "tests/reference_service", "tools/live", "tests/harness"):
        paths.update(p.relative_to(root).as_posix() for p in (root / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})
    paths.add("cmake/LiveTestSupport.cmake")
    paths.update(p.relative_to(root).as_posix() for p in (root / "tests/native/support").iterdir() if p.is_file())
    return sorted(paths)


def validate_dependencies(manifest):
    dependencies = manifest.get("dependencies", [])
    plugins = [entry for entry in dependencies if entry.get("role") == "plugin"]
    metadata = [entry for entry in dependencies if entry.get("role") == "plug_info"]
    native = [entry for entry in dependencies if entry.get("role") == "native_dependency"]
    if len(plugins) != 1 or len(metadata) != 1:
        raise ValueError("resolver/runtime manifest is missing plugin/metadata identity")
    if not native or len(native) + 2 != len(dependencies):
        raise ValueError("resolver/runtime manifest is missing native dependency identity")
    if len({entry["path"] for entry in dependencies}) != len(dependencies):
        raise ValueError("resolver/runtime manifest has duplicate dependency paths")
    trees = manifest.get("dependency_trees", [])
    if len(trees) != len(SDK_TREES) or {entry.get("role") for entry in trees} != SDK_TREES:
        raise ValueError("resolver/runtime manifest is missing SDK identity")
    for entry in trees:
        if tree_hash(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("native resolver/runtime SDK differs; rebuild required")
    plugin = json.loads(Path(metadata[0]["path"]).read_text())["Plugins"]
    if len(plugin) != 1 or Path(plugin[0]["LibraryPath"]).resolve() != Path(plugins[0]["path"]).resolve():
        raise ValueError("resolver/runtime plugin metadata points to an unrecorded binary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "artifact", "plugin", "plug-info", "usd", "python"):
        parser.add_argument("--" + field, type=Path, required=True)
    for field in ("compiler-id", "compiler-version", "configuration"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--sanitizers", choices=("0", "1"), required=True)
    parser.add_argument("--sanitizer", choices=("none", "address", "thread"))
    parser.add_argument("--dependency", type=Path, action="append", required=True)
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
    dependencies = [("plugin", args.plugin.resolve()), ("plug_info", args.plug_info.resolve())]
    dependencies += [("native_dependency", path) for path in sorted({p.resolve() for p in args.dependency})]
    manifest = {
        "schema_version": 1, "boundary": "resolver_runtime", "configuration": args.configuration,
        "compiler_id": args.compiler_id, "compiler_version": args.compiler_version,
        "sanitizers": args.sanitizers == "1", "sanitizer": kind,
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "inputs": [{"path": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
                   for name in inputs(root)],
        "dependencies": [{"role": role, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for role, path in dependencies],
        "dependency_trees": trees,
    }
    validate_dependencies(manifest)
    destination = Path(str(args.artifact) + ".manifest.json")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
