#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind the Python extension and wrappers to the actual shared plugin and SDKs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live.wire_identity import tree_hash


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(root):
    names = {"cmake/LivePython.cmake", "tools/live/python_identity.py"}
    names.update(p.relative_to(root).as_posix() for p in (root / "tests/native/python").glob("*.py"))
    names.update(p.relative_to(root).as_posix()
                 for p in (root / "source/bindings/python/omni/usd/live").rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})
    names.update(p.relative_to(root).as_posix() for p in (root / "source/kit").rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})
    return {name: digest(root / name) for name in sorted(names)}


def modules(directory, reject_bytecode=True):
    if reject_bytecode and any(p.suffix in {".pyc", ".pyo"} for p in directory.rglob("*")):
        # DONTWRITEBYTECODE alone still permits loading existing unchecked caches.
        raise ValueError("Python binding package contains unrecorded bytecode caches")
    return {p.name: digest(p) for p in sorted(directory.iterdir())
            if p.suffix in {".py", ".pyi"} or p.name == "py.typed"}


def interpreter_identity(path):
    # Preserve a selected venv executable path rather than discarding its environment.
    # -I ignores PYTHONDONTWRITEBYTECODE. This probe must not change an SDK
    # whose headers/libraries were just hashed by the native build manifest.
    result = subprocess.run([str(path), "-I", "-B", "-c",
        "import json,sys,sysconfig; print(json.dumps([sys.version,sysconfig.get_config_var('SOABI')]))"],
        check=True, capture_output=True, text=True, timeout=10)
    return {"path": str(path), "sha256": digest(path), "version_abi": json.loads(result.stdout)}


def kit_configuration(package):
    profile = json.loads((package / "config/live-sdk.json").read_text())
    if (not isinstance(profile, dict) or set(profile) != {"schema_version", "usd_profile"} or
            type(profile["schema_version"]) is not int or profile["schema_version"] != 1 or
            profile["usd_profile"] not in ("stock-25.11", "kit-25.11")):
        raise ValueError("Invalid Kit extension SDK profile")
    files = ("config/extension.toml", "python/omni/kit/live_services/__init__.py",
             "python/omni/kit/live_services/_bootstrap.py")
    for name in files:
        if (package / name).read_bytes() != (ROOT / "source/kit" / name).read_bytes():
            raise ValueError("Kit extension configuration differs from source")
    if any(p.suffix in {".pyc", ".pyo"} for p in (package / "python/omni/kit/live_services").rglob("*")):
        raise ValueError("Kit extension contains unrecorded bytecode caches")
    return {name: digest(package / name) for name in (*files, "config/live-sdk.json")}


def validate(artifact):
    from tools.live.test import native_identity
    artifact = Path(artifact).resolve()
    record = json.loads(Path(str(artifact) + ".manifest.json").read_text())
    if record.get("schema_version") != 1 or record.get("boundary") != "python_adapter":
        raise ValueError("Wrong Python binding artifact identity")
    if record.get("artifact_sha256") != digest(artifact) or record.get("sources") != sources(ROOT):
        raise ValueError("Python binding artifact/sources changed; rebuild required")
    native = native_identity(Path(record["plugin_driver"]), "plugin")
    if record.get("plugin_identity") != native:
        raise ValueError("Python binding plugin dependency changed; rebuild required")
    if record.get("pybind_sha256") != tree_hash(Path(record["pybind_include"])):
        raise ValueError("Python binding pybind SDK changed; rebuild required")
    if record.get("interpreter") != interpreter_identity(Path(record["interpreter"]["path"])):
        raise ValueError("Python binding build interpreter changed; rebuild required")
    if record.get("modules") != modules(artifact.parent):
        raise ValueError("Python binding packaged wrappers changed; rebuild required")
    if record["modules"] != modules(ROOT / "source/bindings/python/omni/usd/live", reject_bytecode=False):
        raise ValueError("Python binding wrappers differ from source")
    plugin = next(Path(e["path"]) for e in native["dependencies"] if e["role"] == "plugin")
    package = plugin.parent.parent
    if artifact.parent != package / "python/omni/usd/live":
        raise ValueError("Python binding is outside its canonical plugin package")
    if record.get("kit_configuration") != kit_configuration(package):
        raise ValueError("Kit extension configuration changed; rebuild required")
    return record


def main():
    from tools.live.test import native_identity
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "artifact", "plugin-driver", "pybind", "interpreter"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    artifact, root = args.artifact.resolve(), args.root.resolve()
    record = {"schema_version": 1, "boundary": "python_adapter", "artifact_sha256": digest(artifact),
              "kit_configuration": kit_configuration(artifact.parents[4]),
              "sources": sources(root), "modules": modules(artifact.parent),
              "plugin_driver": str(args.plugin_driver.resolve()),
              "plugin_identity": native_identity(args.plugin_driver.resolve(), "plugin"),
              "pybind_include": str((args.pybind / "include").absolute()),
              "pybind_sha256": tree_hash(args.pybind / "include"),
              "interpreter": interpreter_identity(args.interpreter.absolute())}
    path = Path(str(artifact) + ".manifest.json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    validate(artifact)


if __name__ == "__main__":
    main()
