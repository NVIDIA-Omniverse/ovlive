# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selected inputs and completed evidence for the bounded headless Kit lane.

The lane qualifies one of two candidate kinds, and the inputs select the kind explicitly. A build-tree package carries
the `_live*.so.manifest.json` sidecar that `tools/live/python_identity.py` binds to its sources, plugin driver, SDK
trees and build interpreter. A released packman package, selected with `--kit-release-package <zip>`, carries none of
those: its provenance is the zip's checksum file and the path-free `release-manifest.json` that
`tools/release/package.py` verifies, and the extracted `--kit-package` folder must hold exactly the manifest's files
with their recorded SHA-256 and size.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_INPUTS = ("kit_executable", "kit_package", "kit_client_extension", "kit_resolver_extension")
INPUTS = (*REQUIRED_INPUTS, "kit_library_dir", "kit_core_extension", "kit_release_package")
BUILD_TREE, RELEASE_PACKAGE = "build-tree", "release-package"
RELEASE_MANIFEST = "release-manifest.json"
# A release package qualifies in Kit only when it binds to Kit's USD.
KIT_USD_PROFILE = "kit-25.11"
TESTS = {
    "test_kit_process.KitProcessTests.test_kit_direct_live_root_issues_host_edits": 1,
    "test_kit_process.KitProcessTests.test_kit_healthy_layer_progresses_during_recovery_with_stalled_diagnostics": 1,
    "test_kit_process.KitProcessTests.test_kit_preserves_unknown_and_queued_edits_after_lost_response": 1,
    "test_kit_process.KitProcessTests.test_two_kit_stages_converge_after_seeded_overlapping_edits": 2,
    "test_kit_process.KitProcessTests.test_kit_stage_resumes_after_load_balancer_backend_rotation": 1,
}
PROFILE = "headless-overlay-reference-v2"
SETTINGS = {"/exts/omni.usd/locking/earlyRelease": True,
            "/exts/omni.usd/locking/onClose": True,
            "/app/useFabricSceneDelegate": False}
RUNTIME_KEYS = ("stage_root", "python_version", "python_optimize", "dont_write_bytecode", "settings", "driver_sha256",
                "executable_sha256", "extension_configs", "native_libraries", "python_abi", "live_module")


def digest(path):
    path = Path(path)
    if any(part.casefold() == "package-info.yaml" for part in (*path.parts, *path.resolve().parts)):
        raise ValueError("package metadata is excluded from Kit qualification inputs")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tree(path):
    """Follow selected overlay links, reject cycles, never read package metadata."""
    value = hashlib.sha256()
    files = size = entries = 0

    def visit(directory, relative, parents):
        nonlocal files, size, entries
        resolved = directory.resolve(strict=True)
        if resolved in parents:
            raise ValueError("cycle in Kit input tree")
        for child in sorted(directory.iterdir()):
            if child.name.casefold() == "package-info.yaml":
                continue
            if any(part.casefold() == "package-info.yaml" for part in child.resolve(strict=True).parts):
                continue
            entries += 1
            if entries > 20000:
                raise ValueError("Kit input tree exceeds 20000 entries")
            name = relative / child.name
            if child.is_dir():
                visit(child, name, parents | {resolved})
            elif child.is_file():
                record = [name.as_posix(), str(child.resolve(strict=True)), digest(child)]
                value.update(json.dumps(record, separators=(",", ":")).encode() + b"\n")
                files += 1
                size += child.stat().st_size
            else:
                raise ValueError("unsupported or missing Kit input entry")

    visit(path, Path(), set())
    if not files:
        raise ValueError("empty Kit input tree")
    return dict(path=str(path), resolved=str(path.resolve()), sha256=value.hexdigest(), files=files, bytes=size)


def _elf_machine(path):
    with Path(path).open("rb") as stream:
        header = stream.read(20)
    if len(header) < 20 or header[:4] != b"\x7fELF":
        raise ValueError("Kit release package binary is not ELF: " + Path(path).name)
    return int.from_bytes(header[18:20], "little")


def release_candidate(archive_path, package):
    """Verify the extracted release `package` folder against the packman zip `archive_path` it was unpacked from.

    `tools/release/package.py`'s `check` binds the zip to its `.sha256` checksum file and every member to the release
    manifest; the folder must then hold exactly that manifest and its files, byte for byte. The manifest must name a
    clean Kit USD package of this host's platform whose Python extension carries the CPython tag of the packaged
    Python component. `validate_runtime` checks the module Kit loads against the result.
    """
    from tools.live import build, ci, platforms
    from tools.release import package as release
    archive_path, package = Path(archive_path), Path(package)
    if not archive_path.is_file():
        raise ValueError("Kit release package zip is missing: " + str(archive_path))
    sha256 = digest(archive_path)
    try:
        record = release.check(archive_path)
    except zipfile.BadZipFile as error:
        raise ValueError("Kit release package is not a readable zip: " + str(error)) from None
    host = platforms.host()
    if record["usd_profile"] != KIT_USD_PROFILE:
        raise ValueError(f"Kit release package is built for USD profile {record['usd_profile']}, not {KIT_USD_PROFILE}")
    if record["platform"] != host.name:
        raise ValueError(f"Kit release package is built for {record['platform']}, not this {host.name} host")
    if record.get("dirty") is not False:
        raise ValueError("Kit release package was built from a modified checkout")
    prefix = record["extension_folder"] + "/"
    expected = {name[len(prefix):]: entry for name, entry in record["files"].items()}
    actual = set()
    for directory, directories, files in os.walk(package):
        for name in directories:
            if (Path(directory) / name).is_symlink():
                raise ValueError("Extracted release package holds a directory link: " + name)
        for name in files:
            if name.casefold() == "package-info.yaml":
                continue
            path = Path(directory) / name
            if not path.is_file():
                raise ValueError("Extracted release package holds a non-file entry: " + name)
            actual.add(path.relative_to(package).as_posix())
    wanted = {RELEASE_MANIFEST, *expected}
    if actual != wanted:
        raise ValueError(f"Extracted release package files differ from its manifest: "
                         f"extra={sorted(actual - wanted)}, missing={sorted(wanted - actual)}")
    for name, entry in sorted(expected.items()):
        path = package / name
        if path.stat().st_size != entry["size"] or digest(path) != entry["sha256"]:
            raise ValueError("Extracted release package member differs from its manifest: " + name)
    with (package / RELEASE_MANIFEST).open("rb") as stream:
        extracted = json.loads(stream.read(16 * 1024 * 1024 + 1), object_pairs_hook=ci.unique_object)
    if extracted != record:
        raise ValueError("Extracted release-manifest.json differs from the release package zip's")
    python = [entry for entry in record["components"] if isinstance(entry, dict) and entry.get("name") == "Python"]
    version = python[0].get("version") if len(python) == 1 else None
    adapter = build.python_extension(record["platform"])
    if (not isinstance(version, str) or not record["package"]["name"].endswith("_py_" + version) or
            not host.python_extension_suffix.startswith(".cpython-" + version.replace(".", "") + "-")):
        raise ValueError(f"Kit release package Python ABI is not CPython {version} of {host.name}")
    for name in (adapter, build.library(record["platform"])):
        if _elf_machine(package / name) != host.machine_code:
            raise ValueError("Kit release package binary is built for another machine: " + name)
    profile = json.loads((package / "config/live-sdk.json").read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("usd_profile") != record["usd_profile"]:
        raise ValueError("Kit release package SDK profile differs from its manifest")
    if digest(archive_path) != sha256:
        raise ValueError("Kit release package zip changed during verification")
    return dict(archive=dict(path=str(archive_path.absolute()), resolved=str(archive_path.resolve()), sha256=sha256),
                abi_version=int(record["abi_version"], 16), python_extension=adapter, manifest=record)


def selection(args):
    return {name: str(getattr(args, name).absolute()) if getattr(args, name) is not None else None
            for name in INPUTS}


def snapshot(args):
    if not sys.platform.startswith("linux"):
        raise ValueError("Kit reference qualification requires Linux")
    paths = selection(args)
    for name in REQUIRED_INPUTS:
        if paths[name] is None:
            raise ValueError("--" + name.replace("_", "-") + " is required for kit-reference")
    executable = Path(paths["kit_executable"])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("Kit executable is missing or not executable")
    package = Path(paths["kit_package"])
    extensions = list((package / "python/omni/usd/live").glob("_live*.so"))
    if len(extensions) != 1:
        raise ValueError("Kit package needs one native Python adapter")
    adapter = extensions[0]
    if paths["kit_release_package"] is None:
        if (package / RELEASE_MANIFEST).exists():
            raise ValueError("--kit-package is an extracted release package; select the release-package lane with "
                             "--kit-release-package <zip>")
        from tools.live.python_identity import validate
        lane, candidate = BUILD_TREE, validate(adapter)
    else:
        lane, candidate = RELEASE_PACKAGE, release_candidate(paths["kit_release_package"], package)
        if adapter.relative_to(package).as_posix() != candidate["python_extension"]:
            raise ValueError("Kit release package Python adapter differs from its manifest")
        # Kit runs the resolved folder and derives the extension ID from its name.
        if package.resolve().name != candidate["manifest"]["extension_folder"]:
            raise ValueError("Unpack the Kit release package into a folder named " +
                             candidate["manifest"]["extension_folder"])
    overlays = {}
    for name in ("kit_client_extension", "kit_resolver_extension", "kit_core_extension"):
        if paths[name] is None:
            continue
        path = Path(paths[name])
        if not (path / "config/extension.toml").is_file():
            raise ValueError("Kit extension configuration is missing: " + name)
        overlays[name] = tree(path)
    libraries = tree(Path(paths["kit_library_dir"])) if paths["kit_library_dir"] else None
    configs = {extension: digest(Path(paths[name]) / "config/extension.toml") for extension, name in (
        ("omni_usd_live", "kit_package"), ("omni.client", "kit_client_extension"),
        ("omni.usd_resolver", "kit_resolver_extension"))}
    if paths["kit_core_extension"]:
        configs["omni.usd.core"] = digest(Path(paths["kit_core_extension"]) / "config/extension.toml")
    required_libraries = {}
    for name, relative in (("kit_package", "lib/libomni_usd_live.so"),
                           ("kit_client_extension", "bin/libomni.client.lifecycle.plugin.so"),
                           ("kit_resolver_extension", "lib/libcarb.usdresolver.plugin.so"),
                           ("kit_resolver_extension", "lib/libomni_usd_resolver.so"),
                           ("kit_resolver_extension", "bin/libomni.ext-usd_resolver.plugin.so")):
        library = (Path(paths[name]) / relative).resolve(strict=True)
        required_libraries[str(library)] = digest(library)
    if paths["kit_core_extension"]:
        library = (Path(paths["kit_core_extension"]) / "bin/libomni.tbb.globalcontrol.plugin.so").resolve(strict=True)
        required_libraries[str(library)] = digest(library)
    return dict(schema_version=2, profile=PROFILE, lane=lane, paths=paths, candidate=candidate,
                adapter=dict(path=str(adapter.resolve()), sha256=digest(adapter),
                             ext_suffix=adapter.name.removeprefix("_live")),
                overlays=overlays,
                extension_configs=configs, required_libraries=required_libraries,
                library_dir=libraries, executable=dict(path=str(executable), resolved=str(executable.resolve()),
                                                      sha256=digest(executable)))


def reference_environment(environment, identity):
    isolation = dict(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PXR_PLUGINPATH_NAME="",
                     PXR_AR_DEFAULT_SEARCH_PATH="", AR_DEFAULT_RESOLVER="")
    if identity["lane"] == RELEASE_PACKAGE:
        # The release-package preflight runs no build-tree driver, so the reference service needs no SDK library
        # path; Kit supplies its own USD and Python.
        return dict({key: value for key, value in environment.items() if key != "LD_LIBRARY_PATH"}, **isolation)
    sdk = identity["candidate"]["plugin_identity"]["dependency_trees"]
    libraries = [entry["path"] for entry in sdk if entry["role"] in {"usd_lib", "python_lib"}]
    return dict(environment, LD_LIBRARY_PATH=os.pathsep.join(libraries), **isolation)


def arguments(identity):
    return [value for name, path in identity["paths"].items() if path is not None
            for value in ("--" + name.replace("_", "-"), path)]


def validate_runtime(runtime, identity):
    expected = {test.rsplit(".", 1)[-1] + "-kit-" + str(index)
                for test, count in TESTS.items() for index in range(count)}
    if not isinstance(runtime, dict) or set(runtime) != expected:
        raise ValueError("Kit replay participant inventory differs")
    for participant, probe in runtime.items():
        if (set(probe) != set(RUNTIME_KEYS) or type(probe["python_optimize"]) is not int or
                probe['stage_root'] != ('live' if participant == 'test_kit_direct_live_root_issues_host_edits-kit-0' else 'composed') or
                probe["python_optimize"] != 0 or probe["dont_write_bytecode"] is not True or
                json.dumps(probe["settings"], sort_keys=True) != json.dumps(SETTINGS, sort_keys=True) or
                probe["extension_configs"] != identity["extension_configs"] or
                probe["executable_sha256"] != identity["executable"]["sha256"] or
                probe["driver_sha256"] != digest(ROOT / "tests/native/python/kit_process.py") or
                not isinstance(probe["python_version"], str) or not probe["python_version"]):
            raise ValueError("Kit runtime profile or selected configuration differs")
        module, abi = probe["live_module"], probe["python_abi"]
        if (not isinstance(module, dict) or not isinstance(abi, dict) or
                module.get("path") != identity["adapter"]["path"] or
                module.get("sha256") != identity["adapter"]["sha256"] or
                abi.get("ext_suffix") != identity["adapter"]["ext_suffix"] or type(module.get("abi_version")) is not int):
            raise ValueError("Kit loaded another Python adapter or runs an interpreter of another ABI")
        if identity["lane"] == RELEASE_PACKAGE and module["abi_version"] != identity["candidate"]["abi_version"]:
            raise ValueError("Kit-loaded runtime reports another ABI version than the release manifest")
        libraries = probe["native_libraries"]
        if not isinstance(libraries, dict) or not libraries:
            raise ValueError("Kit replay lacks mapped library identity")
        if any(libraries.get(path) != value for path, value in identity["required_libraries"].items()):
            raise ValueError("Kit did not map required candidate/lifecycle/resolver libraries")
        for path, expected in libraries.items():
            if not Path(path).is_absolute() or digest(Path(path)) != expected:
                raise ValueError("Kit replay mapped library differs")


def release_sha256(identity):
    """The SHA-256 of the release package zip a release-package identity validated; None for a build tree."""
    return identity["candidate"]["archive"]["sha256"] if identity["lane"] == RELEASE_PACKAGE else None


def read_json(path):
    with Path(path).open("rb") as stream:
        content = stream.read(16 * 1024 * 1024 + 1)
    if len(content) > 16 * 1024 * 1024:
        raise ValueError("Kit evidence exceeds 16 MiB")
    return json.loads(content)


def participants(directory, identity, *, require_clean=True):
    """Retain runtime identity even for a failed test, after every child started."""
    result = {}
    expected = {test.rsplit(".", 1)[-1] + "-kit-" + str(index)
                for test, count in TESTS.items() for index in range(count)}
    actual = {path.name for path in directory.glob("test_*-kit-*") if path.is_dir()}
    if actual != expected:
        raise ValueError("Kit participant directory inventory differs")
    for test, count in TESTS.items():
        for index in range(count):
            name = test.rsplit(".", 1)[-1] + "-kit-" + str(index)
            child = directory / name
            process = read_json(child / "process.json")
            probe = read_json(child / "probe.json")
            if require_clean and (process.get("outcome") != "passed" or process.get("returncode") != 0 or
                    process.get("cleanup_complete") is not True or process.get("output_truncated") is not False or
                    probe.get("passed") is not True or
                    probe.get("executable_sha256") != identity["executable"]["sha256"] or
                    probe.get("driver_sha256") != digest(ROOT / "tests/native/python/kit_process.py")):
                raise ValueError("incomplete Kit participant or changed executable/driver: " + name)
            result[name] = {key: probe[key] for key in RUNTIME_KEYS}
    validate_runtime(result, identity)
    return result


def completion(directory, identity, source, seed):
    summary = read_json(directory / "kit-reference-result.json")
    if (summary.get("schema_version") != 1 or summary.get("boundary") != "kit-reference" or
            summary.get("profile") != PROFILE or summary.get("passed") is not True or
            summary.get("evidence_errors") != [] or
            summary.get("test_ids") != sorted(TESTS) or
            any(type(summary.get(key)) is not int or summary[key] != value for key, value in {
                "discovered": len(TESTS), "tests_run": len(TESTS), "skipped": 0, "failures": 0,
                "errors": 0, "expected_failures": 0, "unexpected_successes": 0, "seed": seed}.items()) or
            summary.get("source_fingerprint") != source or summary.get("final_source_fingerprint") != source or
            summary.get("inputs_unchanged") is not True or summary.get("kit_identity") != identity or
            summary.get("kit_lane") != identity["lane"] or
            summary.get("release_package_sha256") != release_sha256(identity)):
        raise ValueError("Kit reference completion evidence is missing, stale or incomplete")
    if summary.get("participants") != participants(directory, identity):
        raise ValueError("Kit participant evidence differs from completed summary")
    return summary
