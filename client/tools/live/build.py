#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, offline native USD 25.11 / Python 3.12 Live Services build/staging.

plan(args) validates inputs without running tools. execute(plan) configures/builds/
installs, then publishes a fresh owned package. No dependency fetching is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live import platforms  # noqa: E402

USD_PROFILES = {'stock-25.11': 'pxr', 'kit-25.11': 'ov'}
MANIFEST = "package-manifest.json"
PYTHON_PACKAGE = "python/omni/usd/live/"
SAMPLE_NAMES = ("live_diagnostics_sample", "live_participant_sample")
# The package members whose names every platform shares.
PAYLOAD_FILES = (
    "resources/plugInfo.json", "config/extension.toml", "config/live-sdk.json",
    "include/omni/usd/live/Live.h", "include/omni/usd/live/LiveUsd.h",
    "share/omni_usd_live/LICENSE", "share/omni_usd_live/NOTICE", "share/omni_usd_live/THIRD_PARTY_NOTICES.md",
    "share/omni_usd_live/THIRD_PARTY_LICENSES.txt", "share/omni_usd_live/BOM.json",
    "share/omni_usd_live/provenance/live-proto.md",
    "share/omni_usd_live/provenance/wire-schema/provenance.json",
    "share/omni_usd_live/provenance/wire-schema/UPSTREAM-NOTICE.txt",
    "share/omni_usd_live/provenance/flatbuffers/LICENSE.txt",
    "share/omni_usd_live/provenance/flatbuffers/provenance.json",
    PYTHON_PACKAGE + "__init__.py", PYTHON_PACKAGE + "_host.py",
    PYTHON_PACKAGE + "_kit_host.py", PYTHON_PACKAGE + "_kit_sessions.py", PYTHON_PACKAGE + "_kit_root.py",
    "python/omni/kit/live_services/__init__.py", "python/omni/kit/live_services/_bootstrap.py",
    PYTHON_PACKAGE + "__init__.pyi", PYTHON_PACKAGE + "py.typed", PYTHON_PACKAGE + "sample.py",
)


def python_extension(name):
    """The package path of the Python extension module built for platform `name`."""
    return PYTHON_PACKAGE + "_live" + platforms.get(name).python_extension_suffix


def library(name):
    """The plugin library of platform `name`. On Windows it sits beside the sample executables, where the loader
    finds a DLL an executable imports; Unix binaries find it through their runtime search paths."""
    return {"pe": "bin/omni_usd_live.dll", "elf": "lib/libomni_usd_live.so",
            "macho": "lib/libomni_usd_live.dylib"}[platforms.get(name).binary_format]


PLUG_INFO = "source/live/scene/plugin/plugInfo.json.in"


def plug_info(root, name):
    """The plugInfo.json the build configures for platform `name` from the tracked template: the template with its
    plugin library path."""
    return (Path(root) / PLUG_INFO).read_bytes().replace(b"@LIVE_PLUGIN_LIBRARY@", library(name).encode())


def samples(name):
    """The sample executables of platform `name`."""
    suffix = ".exe" if platforms.get(name).binary_format == "pe" else ""
    return tuple(f"bin/{sample}{suffix}" for sample in SAMPLE_NAMES)


def packaged(name):
    """Whether platform `name` has a defined package layout; the ELF, PE and Mach-O platforms do."""
    return platforms.get(name).binary_format in ("elf", "pe", "macho")


def package_files(name):
    """The package payload of platform `name`; a platform without a defined package is an error. On Windows it also
    holds the plugin's import library, which a C ABI consumer links."""
    if not packaged(name):
        raise ValueError(f"{name} has no package layout yet (open item platform, #21)")
    imports = ("lib/omni_usd_live.lib",) if platforms.get(name).binary_format == "pe" else ()
    return (library(name), *imports, *PAYLOAD_FILES, *samples(name), python_extension(name))


def native_files(name):
    """The package members in the platform's native binary format: the library, the samples and the extension."""
    return (library(name), *samples(name), python_extension(name))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# The installed notice names map to their source paths relative to the client root.
NOTICES = {"LICENSE": "../LICENSE", "NOTICE": "NOTICE",
           "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md",
           "THIRD_PARTY_LICENSES.txt": "THIRD_PARTY_LICENSES.txt", "BOM.json": "BOM.json"}


def _inputs(root):
    """The build input hashes of the client at `root`, keyed by client-relative path and, for NOTICES, by name."""
    files = {root / "CMakeLists.txt", root / "tools/live/build.py"}
    for relative in ("cmake", "source/live", "source/kit", "include/omni/usd/live",
                     "source/bindings/python/omni/usd/live"):
        files.update(p for p in (root / relative).rglob("*")
                     if p.is_file() and p.name != "PACKAGE-INFO.yaml" and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo"))
    inputs = {p.relative_to(root).as_posix(): _hash(p) for p in files}
    inputs.update({name: _hash(root / source) for name, source in NOTICES.items()})
    return dict(sorted(inputs.items()))


def _tree_hash(root):
    """Hash whitelisted SDK trees, including symlinked headers/libraries, never package manifests."""
    digest = hashlib.sha256()

    def visit(directory, ancestors):
        resolved = directory.resolve(strict=True)
        if resolved in ancestors:
            raise ValueError(f"Cycle in SDK input tree: {directory}")
        for path in sorted(directory.iterdir()):
            if path.name == "__pycache__" or (path.is_file() and path.suffix in {".pyc", ".pyo"}):
                continue
            if path.name == "PACKAGE-INFO.yaml" or path.resolve(strict=True).name == "PACKAGE-INFO.yaml":
                continue
            if path.is_dir():
                visit(path, ancestors | {resolved})
            elif path.is_file():
                digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
                digest.update(bytes.fromhex(_hash(path)))
            else:
                raise ValueError(f"Unsupported SDK input: {path}")

    visit(root, set())
    return digest.hexdigest()


def _support_identity(support):
    """Recompute content identity only for explicitly selected tools and SDK trees."""
    tools, sdks = support.get("tools", {}), support.get("sdks", {})
    if (not isinstance(tools, dict) or not isinstance(sdks, dict) or
            support.get("config") not in ("debug", "release") or
            set(tools) not in ({"cmake", "ninja", "cc", "cxx", "protoc", "grpc_cpp_plugin", "python"},
                           {"cmake", "make", "cc", "cxx", "protoc", "grpc_cpp_plugin", "python"}) or
            set(sdks) != {"usd_root", "python_root", "dependency_root", "pybind11_root"}):
        raise ValueError("Incomplete build support inputs")
    usd_profile = support.get("usd_profile", "stock-25.11")
    if not isinstance(usd_profile, str) or usd_profile not in USD_PROFILES:
        raise ValueError("Unsupported USD profile: " + str(usd_profile))
    files = {name: {"path": value, "sha256": _hash(_path(value, executable=True))}
             for name, value in sorted(tools.items())}
    trees = {}
    for name in sorted(sdks):
        base = Path(sdks[name])
        if name == "dependency_root":
            directories = [f"{package}-{support['config']}/{part}"
                           for package in ("grpc", "protobuf", "abseil", "re2", "c-ares", "openssl", "zlib")
                           for part in ("include", "lib")]
        elif name == "pybind11_root":
            directories = ["include"]
        else:
            directories = ["include", "lib"]
        for part in directories:
            path = base / part
            _path(path, directory=True)
            trees[f"{name}/{part}"] = {"path": str(path), "sha256": _tree_hash(path)}
    return {"tools": files, "trees": trees, "usd_profile": usd_profile}


def _path(value, directory=False, executable=False, preserve_selection=False):
    selected = Path(value).expanduser().absolute()
    path = selected.resolve(strict=True)
    if (directory and not path.is_dir()) or (not directory and not path.is_file()):
        raise ValueError(f"Wrong input type: {path}")
    if executable and not os.access(path, os.X_OK):
        raise ValueError(f"Tool is not executable: {path}")
    # CMake list syntax cannot represent these input paths faithfully.
    if any(c in str(candidate) for candidate in (selected, path) for c in (";", "\n", "\r")):
        raise ValueError(f"Unsupported input path: {selected}")
    # SDK/tool selections can be symlinks maintained by the dependency manager.
    # Record that selection, not just today's destination, so later validation
    # hashes its current content and rejects a retargeted dependency.
    return selected if preserve_selection else path


def _define(path, name, expected):
    text = path.read_text(encoding="utf-8")
    if not re.search(r"^\s*#\s*define\s+" + re.escape(name) + r"\s+" + str(expected) + r"\s*$", text, re.M):
        raise ValueError(f"Unsupported SDK profile: {path} must define {name} {expected}")


def _sibling_resolver(target):
    sibling = target.parent / "resolver/resources/plugInfo.json"
    if sibling.exists() or sibling.is_symlink():
        raise ValueError(f"Sibling resolver metadata would co-register with live://: {sibling}; use a fresh output tree")


def _owned_target(target):
    if target.is_symlink():
        raise ValueError("Refusing symlink package destination")
    if target.exists():
        if not target.is_dir():
            raise ValueError("Refusing non-directory package destination")
        old = json.loads((target / MANIFEST).read_text())
        if old.get("owner") != "native-live-build" or old.get("schema_version") != 1:
            raise ValueError("Refusing to replace an unrecognized package; remove it explicitly")


def python_sdk(root, name):
    """The header, interpreter and library paths of an SDK Python 3.12 on platform `name`: a Windows Python package
    keeps its headers in include, its import library in libs and its interpreter at the root."""
    if platforms.get(name).binary_format == "pe":
        return dict(patchlevel=root / "include/patchlevel.h", interpreter=root / "python.exe",
                    library=root / "libs/python312.lib")
    suffix = "dylib" if platforms.get(name).binary_format == "macho" else "so"
    return dict(patchlevel=root / "include/python3.12/patchlevel.h", interpreter=root / "bin/python3.12",
                library=root / f"lib/libpython3.12.{suffix}")


def _usd_libraries(root, profile, platform="linux-x86_64"):
    if not isinstance(profile, str) or profile not in USD_PROFILES:
        raise ValueError("Unsupported USD profile: " + str(profile))
    header_path = root / "include/pxr/pxr.h"
    _define(header_path, "PXR_VERSION", 2511)
    header = header_path.read_text()
    namespace = USD_PROFILES[profile]
    for name, value in (("PXR_NS", namespace),
                        ("PXR_INTERNAL_NS", namespace + "Internal_v0_25_11__pxrReserved__")):
        if not re.search(r'^\s*#\s*define\s+' + name + r'\s+' + re.escape(value) + r'\s*$', header, re.M):
            raise ValueError(f"USD header does not match {profile}: {name}")
    if not re.search(r'^\s*#\s*define\s+PXR_PYTHON_SUPPORT_ENABLED\s*$', header, re.M):
        raise ValueError("The selected USD SDK must enable Python")
    libraries = (("ov_25.11usd_ms", "tbb") if profile == "kit-25.11" else
                 ("usd_sdf", "usd_ar", "usd_plug", "usd_usdGeom", "usd_vt", "usd_gf", "usd_tf",
                  "usd_arch", "usd_trace", "usd_work", "usd_python", "usd_js", "tbb"))
    # A Windows consumer links the import library; the DLL is a runtime dependency.
    windows = platforms.get(platform).binary_format == "pe"
    suffix = "dylib" if platforms.get(platform).binary_format == "macho" else "so"
    for library in libraries:
        _path(root / "lib" / (f"{library}.lib" if windows else f"lib{library}.{suffix}"))
    return libraries


def plan(args):
    """Return a JSON-serializable validated command plan; do not create/run anything."""
    if platforms.get(args.platform) != platforms.host():
        raise ValueError(f"Live Services builds natively: {args.platform} does not match this host")
    package_files(args.platform)
    root = _path(args.root, directory=True)
    _path(root / "CMakeLists.txt")
    config = args.config
    if config not in ("debug", "release"):
        raise ValueError("Supported configurations are debug and release")
    ninja, make = getattr(args, "ninja", None), getattr(args, "make", None)
    if bool(ninja) == bool(make):
        raise ValueError("Select exactly one explicit --ninja or --make executable")
    parallel = getattr(args, "parallel", 2)
    if not isinstance(parallel, int) or not 1 <= parallel <= 64:
        raise ValueError("--parallel must be between 1 and 64 (default 2)")
    generator_tool = "ninja" if ninja else "make"
    tools = {key: str(_path(getattr(args, key), executable=True, preserve_selection=True))
             for key in ("cmake", generator_tool, "cc", "cxx", "protoc", "grpc_cpp_plugin")}
    sdks = {key: _path(getattr(args, key), directory=True, preserve_selection=True) for key in
            ("usd_root", "python_root", "dependency_root", "pybind11_root")}
    usd_profile = getattr(args, "usd_profile", "stock-25.11")
    _usd_libraries(sdks["usd_root"], usd_profile, args.platform)
    python = python_sdk(sdks["python_root"], args.platform)
    _define(python["patchlevel"], "PY_MAJOR_VERSION", 3)
    _define(python["patchlevel"], "PY_MINOR_VERSION", 12)
    tools["python"] = str(_path(python["interpreter"], executable=True, preserve_selection=True))
    _path(sdks["pybind11_root"] / "include/pybind11/pybind11.h")
    for part, version in (("MAJOR", 2), ("MINOR", 11), ("PATCH", 1)):
        _define(sdks["pybind11_root"] / "include/pybind11/detail/common.h", "PYBIND11_VERSION_" + part, version)
    for key in sdks:
        if key != "dependency_root":
            _path(sdks[key] / "include", directory=True)
    for package in ("grpc", "protobuf", "abseil", "re2", "c-ares", "openssl", "zlib"):
        for directory in ("include", "lib"):
            _path(sdks["dependency_root"] / f"{package}-{config}" / directory, directory=True)
    _path(python["library"])
    build = Path(args.build_dir or root.parent / ".cache/live-repo" / args.platform / config).absolute()
    target = Path(args.target_dir).absolute() / "usd/omniverse/live"
    if build.is_symlink() or target.is_symlink():
        raise ValueError("Build/package directories must not be symlinks")
    _sibling_resolver(target)
    _owned_target(target)
    cmake_config = config.capitalize()
    settings = {
        "CMAKE_BUILD_TYPE": cmake_config, "CMAKE_MAKE_PROGRAM": tools[generator_tool],
        "CMAKE_C_COMPILER": tools["cc"], "CMAKE_CXX_COMPILER": tools["cxx"],
        "CMAKE_INSTALL_LIBDIR": "lib", "CMAKE_INSTALL_BINDIR": "bin", "CMAKE_INSTALL_INCLUDEDIR": "include",
        "CMAKE_INSTALL_DATADIR": "share",
        "BUILD_TESTING": "OFF", "LIVE_ENABLE_SANITIZERS": "OFF",
        # The package binaries link with their install RPATH, so no build directory reaches their bytes.
        "CMAKE_BUILD_WITH_INSTALL_RPATH": "ON",
        "LIVE_BUILD_RESOLVER_PROBE": "OFF", "LIVE_BUILD_RESOLVER_RUNTIME": "OFF",
        "LIVE_DEPENDENCY_CONFIG": config, "LIVE_PROTOC": tools["protoc"], "LIVE_GRPC_CPP_PLUGIN": tools["grpc_cpp_plugin"],
        "LIVE_PYTHON_EXECUTABLE": tools["python"],
        "LIVE_USD_PROFILE": usd_profile,
    }
    for flag in ("SCENE_CORE", "WIRE", "USD_WIRE", "WIRE_SCENE", "GRPC_RUNTIME", "SCENE_TRANSPORT", "PLUGIN", "PYTHON"):
        settings[f"LIVE_BUILD_{flag}"] = "ON"
    for key, value in sdks.items():
        settings["LIVE_USD_PYTHON_ROOT" if key == "python_root" else "LIVE_" + key.upper()] = str(value)
    configure = [tools["cmake"], "-S", str(root), "-B", str(build), "-G", "Ninja" if ninja else "Unix Makefiles"]
    configure += [f"-D{key}={value}" for key, value in sorted(settings.items())]
    return {"root": str(root), "platform": args.platform, "config": config, "usd_profile": usd_profile, "build_dir": str(build),
            "target": str(target), "tools": tools, "sdks": {k: str(v) for k, v in sdks.items()},
            "commands": [configure, [tools["cmake"], "--build", str(build), "--parallel", str(parallel), "--target", "omni_usd_live_python", "live_diagnostics_sample", "live_participant_sample"],
                         [tools["cmake"], "--install", str(build), "--prefix", "<fresh-install-prefix>"]]}


def _native(path, name):
    """Reject a file that is not a native binary of platform `name`: a 64-bit little-endian ELF, Mach-O or PE image whose
    header names the platform's machine."""
    entry = platforms.get(name)
    machine = entry.machine_code.to_bytes(4 if entry.binary_format == "macho" else 2, "little")
    with path.open("rb") as stream:
        header = stream.read(64)
        if entry.binary_format == "elf":
            valid = len(header) >= 20 and header[:6] == b"\x7fELF\x02\x01" and header[18:20] == machine
        elif entry.binary_format == "macho":
            valid = len(header) >= 32 and header[:4] == b"\xcf\xfa\xed\xfe" and header[4:8] == machine
        else:
            valid = False
            if len(header) == 64 and header[:2] == b"MZ":
                stream.seek(int.from_bytes(header[0x3C:0x40], "little"))
                signature = stream.read(6)
                valid = signature[:4] == b"PE\0\0" and signature[4:6] == machine
    if not valid:
        raise ValueError(f"Not a {name} {entry.binary_format.upper()} artifact: {path}")


def _validate_layout(path, name, *, exact=False):
    """Check the payload of platform `name` at `path`; `exact` also rejects any member it does not list."""
    files = package_files(name)
    if exact:
        expected = set(files)
        actual = {p.relative_to(path).as_posix() for p in path.rglob("*")
                  if p.is_file() or p.is_symlink()}
        if actual != expected or any(p.is_symlink() for p in path.rglob("*")):
            raise ValueError(f"Unexpected or missing installed members: "
                             f"extra={sorted(actual - expected)}, missing={sorted(expected - actual)}")
    for relative in files:
        item = path / relative
        if item.is_symlink() or not item.is_file() or (item.stat().st_size == 0 and relative != PYTHON_PACKAGE + "py.typed"):
            raise ValueError(f"Missing or unsafe package member: {item}")
    for relative in native_files(name):
        _native(path / relative, name)
    if platforms.get(name).binary_format in ("elf", "macho"):
        for relative in samples(name):
            if not os.access(path / relative, os.X_OK):
                raise ValueError(f"{relative} is not executable")
    metadata = json.loads((path / "resources/plugInfo.json").read_text())
    plugins = metadata.get("Plugins", [])
    if len(plugins) != 1 or plugins[0].get("Root") != ".." or plugins[0].get("LibraryPath") != library(name):
        raise ValueError("Plugin metadata is not package-relative")
    if plugins[0].get("Name") != "omni_usd_live":
        raise ValueError("Unexpected plugin identity")


def validate_stage(path, root=None, *, expected_platform=None, expected_configuration=None):
    """Reject incomplete, modified, mixed or source-stale packages before packaging."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Unsafe package directory: {path}")
    _sibling_resolver(path)
    record = json.loads((path / MANIFEST).read_text())
    if not isinstance(record, dict):
        raise ValueError("Invalid native package manifest")
    if record.get("schema_version") != 1 or record.get("delegate_version") != 2 or record.get("owner") != "native-live-build":
        raise ValueError("Unrecognized native package manifest")
    if not platforms.supported(record.get("platform")) or record.get("configuration") not in ("debug", "release"):
        raise ValueError("Unsupported package platform/configuration")
    _validate_layout(path, record["platform"])
    files = package_files(record["platform"])
    if ((expected_platform is not None and record["platform"] != expected_platform) or
            (expected_configuration is not None and record["configuration"] != expected_configuration)):
        raise ValueError("Package platform/configuration does not match its destination")
    expected = set(files) | {MANIFEST}
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != expected or any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Unexpected or missing package members")
    if record.get("files") != {name: _hash(path / name) for name in files}:
        raise ValueError("Package member hash mismatch")
    if record.get("inputs") != _inputs(Path(root) if root is not None else ROOT):
        raise ValueError("Package source inputs changed; rebuild Live Services")
    support = record.get("support_inputs", {})
    if not isinstance(support, dict) or support.get("platform") != record["platform"] or support.get("config") != record["configuration"]:
        raise ValueError("Package support inputs do not match the build profile")
    if record.get("support_identity") != _support_identity(support):
        raise ValueError("Package SDK/tools changed; rebuild Live Services")
    return record


def assemble_install(install, package):
    """Assemble all installed payload files without filtering against the allowlist."""
    install, package = Path(install), Path(package)
    shutil.copytree(install / "lib/omni_usd_live", package)
    # On Windows the plugin DLL and the sample executables share bin.
    for directory in ("include", "bin", "share"):
        shutil.copytree(install / directory, package / directory, dirs_exist_ok=True)


def _lock_exclusively(handle):
    """Blocks until this process holds the lock file `handle` exclusively, released when the file closes."""
    if sys.platform == "win32":
        import msvcrt
        handle.seek(0)
        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                pass  # LK_LOCK gives up after ten one-second attempts; keep waiting like flock.
    import fcntl
    fcntl.flock(handle, fcntl.LOCK_EX)


def execute(build_plan):
    """Run the explicit plan and replace only an owned stage; retain old stage on failure."""
    root, build, target = (Path(build_plan[key]) for key in ("root", "build_dir", "target"))
    _sibling_resolver(target)
    _owned_target(target)
    build.mkdir(parents=True, exist_ok=True)
    # Serialize configure/build/stage for this cache across concurrent repo invocations.
    with (build / ".live-build.lock").open("a+") as lock:
        _lock_exclusively(lock)
        identity = {key: build_plan[key] for key in ("root", "platform", "config", "tools", "sdks")}
        identity["usd_profile"] = build_plan.get("usd_profile", "stock-25.11")
        support_identity = _support_identity(identity)
        cache_identity = {**identity, "support_identity": support_identity}
        stamp = build / ".live-build-inputs.json"
        if stamp.exists() and json.loads(stamp.read_text()) != cache_identity:
            raise ValueError("Build cache inputs changed; choose a fresh --build-dir")
        if not stamp.exists() and (build / "CMakeCache.txt").exists():
            raise ValueError("Refusing to reuse an unowned CMake cache")
        stamp.write_text(json.dumps(cache_identity, indent=2) + "\n")
        inputs = _inputs(root)
        # CMake invokes the selected SDK interpreter to determine its ABI. Do
        # not modify that SDK's identity by creating sysconfig bytecode, or let
        # per-user Python packages influence the configured artifact.
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        for command in build_plan["commands"][:2]:
            subprocess.run(command, cwd=root, check=True, env=environment)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".live-stage-", dir=target.parent) as scratch:
            scratch = Path(scratch)
            install, package = scratch / "install", scratch / "package"
            command = build_plan["commands"][2][:-1] + [str(install)]
            subprocess.run(command, cwd=root, check=True, env=environment)
            assemble_install(install, package)
            _validate_layout(package, build_plan["platform"], exact=True)
            if inputs != _inputs(root):
                raise ValueError("Source changed during build; refusing stale publication")
            if support_identity != _support_identity(identity):
                raise ValueError("SDK/tools changed during build; refusing stale publication")
            record = {"schema_version": 1, "delegate_version": 2, "owner": "native-live-build",
                      "platform": build_plan["platform"], "configuration": build_plan["config"],
                      "support_inputs": identity, "support_identity": support_identity, "inputs": inputs,
                      "files": {name: _hash(package / name) for name in package_files(build_plan["platform"])}}
            (package / MANIFEST).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            validate_stage(package, root)
            # Different explicit build caches may target the same package.
            with (target.parent / ".live-package.lock").open("a+") as stage_lock:
                _lock_exclusively(stage_lock)
                _sibling_resolver(target)
                _owned_target(target)
                backup = scratch / "previous"
                # No merge: failure after moving the old owned directory restores it.
                if target.exists():
                    target.rename(backup)
                try:
                    package.rename(target)
                except BaseException:
                    if backup.exists():
                        backup.rename(target)
                    raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", default=str(ROOT))
    result.add_argument("--platform", required=True, choices=tuple(platforms.PLATFORMS))
    result.add_argument("--config", required=True, choices=("debug", "release"))
    result.add_argument("--usd-profile", choices=tuple(USD_PROFILES), default="stock-25.11")
    result.add_argument("--target-dir", required=True)
    result.add_argument("--build-dir")
    for key in ("usd-root", "python-root", "dependency-root", "pybind11-root", "cmake", "cc", "cxx", "protoc", "grpc-cpp-plugin"):
        result.add_argument("--" + key, required=True)
    generator = result.add_mutually_exclusive_group(required=True)
    generator.add_argument("--ninja")
    generator.add_argument("--make")
    result.add_argument("--parallel", type=int, default=2, help="bounded build parallelism, 1 to 64 (default 2)")
    result.add_argument("--dry-run", action="store_true", help="validate inputs and print commands without running them")
    return result


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        build_plan = plan(args)
        if args.dry_run:
            print(json.dumps(build_plan, indent=2, sort_keys=True))
        else:
            execute(build_plan)
        return 0
    except subprocess.CalledProcessError as error:
        print(f"Live Services build command failed ({error.returncode}): {error.cmd}", file=sys.stderr)
        return error.returncode if error.returncode > 0 else 1
    except (OSError, ValueError) as error:
        print(f"Live Services build: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
