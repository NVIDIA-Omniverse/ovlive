#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identity for the exact Sdf scene storage component and explicitly selected scene SDKs."""
import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.live.sdk_tree_identity import tree_hash


def inputs(root):
    paths = {"CMakeLists.txt", "cmake/LiveSceneCore.cmake", "cmake/LiveUsdSdk.cmake", "cmake/LiveSanitizer.cmake", "tools/live/scene_identity.py", "tools/live/sdk_tree_identity.py",
             "tests/native/usd/sdf_scene.cpp"}
    # Conservative closure: include every core header/source, even if a given
    # configuration does not instantiate it. Never walk cached SDK symlinks here.
    for directory in ("include/omni/usd/live", "source/live/support", "source/live/scene", "tests/native/usd"):
        paths.update(p.relative_to(root).as_posix() for p in (root / directory).rglob("*") if p.is_file())
    if (root / "tests/native/support").is_dir():  # A paired benchmark's baseline checkout may predate it.
        paths.add("cmake/LiveTestSupport.cmake")
        paths.update(p.relative_to(root).as_posix() for p in (root / "tests/native/support").iterdir() if p.is_file())
    return sorted(paths)


def sdk_paths(usd, python):
    # Keep the selected aliases used by link/RUNPATH; resolving them before
    # recording would validate an old target after the alias is retargeted.
    trees = [usd / "include", usd / "lib", python / "include"]
    # A Windows Python package links through its import library in libs.
    library = ("libs/python312.lib" if sys.platform == "win32" else
               "lib/libpython3.12.dylib" if sys.platform == "darwin" else "lib/libpython3.12.so")
    return [p.absolute() for p in trees], (python / library).absolute()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "artifact", "usd", "python"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--boundary", choices=("sdf_scene", "scene_benchmark", "usd_edit_boundary_probe"), default="sdf_scene")
    parser.add_argument("--compile-commands", type=Path)
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
    trees, python_library = sdk_paths(args.usd, args.python)
    source_inputs = inputs(root)
    if args.boundary == "usd_edit_boundary_probe":
        source_inputs += ["tools/live/usd_edit_boundary.py", "tools/live/test.py", "tools/live/test_support.py",
                          "tools/live/process_guard.py", "tools/live/sanitizer.py"]
    manifest = {
        "schema_version": 1, "boundary": args.boundary, "configuration": args.configuration,
        "compiler_id": args.compiler_id, "compiler_version": args.compiler_version,
        "sanitizers": args.sanitizers == "1", "sanitizer": kind,
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "inputs": [{"path": p, "sha256": hashlib.sha256((root / p).read_bytes()).hexdigest()} for p in source_inputs],
        "dependencies": [{"path": str(python_library), "sha256": hashlib.sha256(python_library.read_bytes()).hexdigest()}],
        "dependency_trees": [{"path": str(p), "sha256": tree_hash(p)} for p in trees],
    }
    if args.boundary == "scene_benchmark":
        if args.compile_commands is None:
            parser.error("scene benchmark requires compile-command provenance")
        compile_file = args.compile_commands.resolve()
        content = compile_file.read_bytes()
        manifest["dependencies"].append({"path": str(compile_file), "sha256": hashlib.sha256(content).hexdigest()})
        commands = []
        for row in json.loads(content):
            # A Windows command line keeps its backslash paths only under non-POSIX splitting.
            command = row.get("arguments") or shlex.split(row["command"], posix=os.name != "nt")
            if not any("CMakeFiles/live_scene_core.dir/" in token.replace("\\", "/") or
                       "CMakeFiles/live_scene_benchmark.dir/" in token.replace("\\", "/") for token in command):
                continue
            commands.append({"source": Path(row["file"]).resolve().relative_to(root).as_posix(),
                "command_sha256": hashlib.sha256(json.dumps(command).encode()).hexdigest(),
                "measurement_flags": [token for token in command if token.startswith(("-O", "/O", "-g", "/Z", "-std=", "/std:", "-fsanitize=", "/fsanitize="))
                                      or token in ("-DNDEBUG", "/DNDEBUG", "-UNDEBUG", "/UNDEBUG", "-fno-omit-frame-pointer")]})
        if not commands or not any(row["source"] == "tests/native/usd/scene_benchmark.cpp" for row in commands):
            parser.error("missing benchmark compile command")
        manifest["compile_commands"] = sorted(commands, key=lambda row: row["source"])
    destination = Path(str(args.artifact) + ".manifest.json")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
