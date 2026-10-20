#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source, shared plugin, public consumer and explicit SDK provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.live.wire_identity import inputs as wire_inputs, USD_TREES, tree_hash, sdk_identity


def inputs(root):
    paths = set(wire_inputs(root, "scene_transport"))
    paths.update({"cmake/LivePlugin.cmake", "include/omni/usd/live/LiveUsd.h"})
    for directory in ("tests/native/plugin", "tests/native/diagnostics_log", "tests/native/participant", "source/live/samples"):
        paths.update(p.relative_to(root).as_posix() for p in (root / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})
    return sorted(paths)


def validate_dependencies(manifest):
    dependencies = manifest.get("dependencies", [])
    roles = {role: [Path(e["path"]) for e in dependencies if e.get("role") == role]
             for role in ("plugin", "plug_info", "native_dependency", "public_header", "load_probe", "diagnostics_sample", "diagnostic_log_tests", "participant_sample", "participant_tests")}
    if (len(roles["plugin"]) != 1 or len(roles["plug_info"]) != 1 or
            len(roles["participant_tests"]) != 1 or len(roles["participant_sample"]) != 1 or len(roles["diagnostics_sample"]) != 1 or len(roles["diagnostic_log_tests"]) != 1 or
            len(roles["public_header"]) != 2 or len(roles["load_probe"]) != 1 or not roles["native_dependency"] or
            sum(map(len, roles.values())) != len(dependencies) or
            len({e["path"] for e in dependencies}) != len(dependencies)):
        raise ValueError("plugin manifest is missing or duplicates package/dependency identity")
    trees = manifest.get("dependency_trees", [])
    if len(trees) != 1 + len(USD_TREES) or {e.get("role") for e in trees} != {"flatbuffers_include", *USD_TREES}:
        raise ValueError("plugin manifest is missing SDK identity")
    root = Path(__file__).resolve().parents[2]
    for entry in trees:
        path = Path(entry["path"])
        if tree_hash(path) != entry["sha256"]:
            raise ValueError("plugin SDK differs; rebuild required")
        if entry["role"] == "flatbuffers_include" and sdk_identity(root, path) != manifest.get("flatbuffers"):
            raise ValueError("plugin FlatBuffers SDK differs; rebuild required")
    metadata = roles["plug_info"][0]
    plugin = roles["plugin"][0]
    from tools.live import build, platforms
    host = platforms.host().name
    expected = build.plug_info(root, host)
    if metadata.read_bytes() != expected:
        raise ValueError("plugin metadata differs from tracked relative template")
    entry = json.loads(expected)["Plugins"][0]
    package = (metadata.parent / entry["Root"]).resolve()
    if (package / entry["LibraryPath"]).resolve() != plugin.resolve():
        raise ValueError("plugin metadata points to unrecorded library")
    diagnostics, participant = build.samples(host)
    if roles["participant_sample"][0] != package / participant:
        raise ValueError("participant sample outside recorded package")
    if roles["diagnostics_sample"][0] != package / diagnostics:
        raise ValueError("diagnostics sample outside recorded package")
    if {p.name for p in roles["public_header"]} != {"Live.h", "LiveUsd.h"}:
        raise ValueError("plugin public headers missing")
    for path in roles["public_header"]:
        if path != package / "include/omni/usd/live" / path.name:
            raise ValueError("plugin public header outside recorded package")
        if path.read_bytes() != (root / "include/omni/usd/live" / path.name).read_bytes():
            raise ValueError("plugin packaged header differs; rebuild required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "artifact", "plugin", "plug-info", "load-probe", "diagnostics-sample", "diagnostic-log-tests", "participant-sample", "participant-tests", "flatbuffers", "usd", "python"):
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
    trees = [("flatbuffers_include", (args.flatbuffers / "include").resolve())]
    trees += [(role, (getattr(args, role.split("_")[0]) / role.split("_")[1]).resolve()) for role in USD_TREES]
    package = args.plugin.resolve().parent.parent
    dependencies = [("plugin", args.plugin.resolve()), ("plug_info", args.plug_info.resolve()),
                    ("load_probe", args.load_probe.resolve()),
                    ("diagnostics_sample", args.diagnostics_sample.resolve()),
                    ("participant_sample", args.participant_sample.resolve()),
                    ("participant_tests", args.participant_tests.resolve()),
                    ("diagnostic_log_tests", args.diagnostic_log_tests.resolve())]
    dependencies += [("public_header", package / "include/omni/usd/live" / name) for name in ("Live.h", "LiveUsd.h")]
    dependencies += [("native_dependency", p) for p in sorted({p.resolve() for p in args.dependency})]
    manifest = {
        "schema_version": 1, "boundary": "plugin", "configuration": args.configuration,
        "compiler_id": args.compiler_id, "compiler_version": args.compiler_version,
        "sanitizers": args.sanitizers == "1", "sanitizer": kind, "flatbuffers": sdk_identity(root, trees[0][1]),
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "inputs": [{"path": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()} for name in inputs(root)],
        "dependencies": [{"role": role, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for role, path in dependencies],
        "dependency_trees": [{"role": role, "path": str(path), "sha256": tree_hash(path)} for role, path in trees],
    }
    validate_dependencies(manifest)
    path = Path(str(args.artifact) + ".manifest.json")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


if __name__ == "__main__":
    main()
