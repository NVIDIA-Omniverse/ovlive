# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit provisioning and the source and package boundaries of the live build."""
import json
import hashlib
import contextlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

from tools.live import dependencies
from tools.live import build

ROOT = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def not_executable(path):
    """Makes `path` non-executable for the duration: by its mode on POSIX, and on Windows, where os.access reports every
    existing file executable, by reporting this one not."""
    if os.name == "posix":
        mode = path.stat().st_mode
        path.chmod(0o644)
        try:
            yield
        finally:
            path.chmod(mode)
        return
    access, target = os.access, Path(path).resolve()

    def report(candidate, mode, *args, **kwargs):
        if mode & os.X_OK and Path(candidate).resolve() == target:
            return False
        return access(candidate, mode, *args, **kwargs)
    with patch.object(os, "access", side_effect=report):
        yield


PINS = {
    "grpc": "1.65.4-1", "protobuf": "5.26.0-2", "abseil": "20240116.1-1",
    "re2": "20230301-1", "c-ares": "1.34.8-0", "openssl": "3.5.9-0", "zlib": "1.3.2-0",
}


class LiveDependencyGenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.deps = self.root / "deps"
        self.deps.mkdir()
        for kind in ("target", "host"):
            name = f"live-services-{kind}.packman.xml"
            shutil.copyfile(ROOT / "deps" / name, self.deps / name)

    def files(self):
        return {path.relative_to(self.root).as_posix(): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def manifest(self, kind):
        return ET.parse(self.deps / f"live-services-{kind}.generated.packman.xml").getroot()

    def generate(self, target="manylinux_2_35_aarch64", host="manylinux_2_35_x86_64"):
        # Rendering local templates must never resolve packages or run fetch tools.
        with patch("subprocess.run", side_effect=AssertionError("generation attempted a subprocess")), \
             patch("subprocess.Popen", side_effect=AssertionError("generation launched a fetch process")), \
             patch("urllib.request.urlopen", side_effect=AssertionError("generation attempted a network fetch")):
            return dependencies.generate(self.root, target, host)

    def test_exact_pins_target_configuration_and_separate_release_host_tools(self):
        self.generate()
        target = self.manifest("target")
        host = self.manifest("host")
        target_packages = {p.attrib["name"]: p.attrib["version"] for p in target.findall("./dependency/package")}
        host_packages = {p.attrib["name"]: p.attrib["version"] for p in host.findall("./dependency/package")}
        self.assertEqual(target_packages, {name: f"{version}-manylinux_2_35_aarch64-static-${{config}}"
                                           for name, version in PINS.items()})
        self.assertEqual(host_packages, {name: f"{PINS[name]}-manylinux_2_35_x86_64-static-release"
                                        for name in ("grpc", "protobuf")})
        self.assertEqual(len(target.findall("dependency")), 7)
        self.assertEqual(len(host.findall("dependency")), 2)
        self.assertFalse(target.findall("import") or host.findall("import"))
        self.assertEqual({d.attrib["linkPath"] for d in target.findall("dependency")},
                         {f"../_build/target-deps/{name}-${{config}}" for name in PINS})
        self.assertEqual({d.attrib["linkPath"] for d in host.findall("dependency")},
                         {f"../_build/host-deps/live-services/{name}-release" for name in ("grpc", "protobuf")})
        metadata = json.loads((self.deps / "live-services-build.json").read_text())
        self.assertEqual(metadata, {"schema_version": 1, "enabled": True,
                                    "target_platform_abi": "manylinux_2_35_aarch64",
                                    "host_platform_abi": "manylinux_2_35_x86_64"})
        self.assertNotIn("${live_services_", "".join(path.read_text() for path in self.deps.glob("*.generated.packman.xml")))

    def test_generation_is_deterministic_and_templates_are_unchanged(self):
        templates = {path.name: path.read_bytes() for path in self.deps.glob("*.packman.xml")}
        self.generate()
        first = self.files()
        self.generate()
        self.assertEqual(self.files(), first)
        for name, content in templates.items():
            self.assertEqual((self.deps / name).read_bytes(), content)

    def test_cli_renders_requested_abis_into_the_given_root(self):
        with patch("subprocess.run", side_effect=AssertionError("generation attempted a subprocess")), \
             patch("subprocess.Popen", side_effect=AssertionError("generation launched a fetch process")), \
             patch("urllib.request.urlopen", side_effect=AssertionError("generation attempted a network fetch")):
            self.assertEqual(dependencies.main(["--root", str(self.root), "--target-abi", "manylinux_2_35_aarch64",
                                                "--host-abi", "manylinux_2_35_x86_64"]), 0)
        for kind, abi in (("target", "manylinux_2_35_aarch64"), ("host", "manylinux_2_35_x86_64")):
            versions = [p.attrib["version"] for p in self.manifest(kind).findall("./dependency/package")]
            self.assertTrue(versions)
            self.assertTrue(all(f"-{abi}-static-" in version for version in versions), versions)
        metadata = json.loads((self.deps / "live-services-build.json").read_text())
        self.assertEqual((metadata["target_platform_abi"], metadata["host_platform_abi"]),
                         ("manylinux_2_35_aarch64", "manylinux_2_35_x86_64"))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            dependencies.main(["--root", str(self.root), "--target-abi", "manylinux_2_35_x86_64"])

    def test_missing_and_invalid_abi_fail_before_any_generated_file_changes(self):
        self.generate()
        before = self.files()
        invalid = (None, "", " ", "-leading", ".leading", "../escape", "linux/x86_64", "a\\b",
                   "${platform}", "linux x86_64", "a\nb", "a\x00b", "a;fetch", "$(fetch)", "é")
        for token in invalid:
            for position in ("target", "host"):
                with self.subTest(token=repr(token), position=position):
                    values = {"target": "manylinux_2_35_x86_64", "host": "manylinux_2_35_x86_64"}
                    values[position] = token
                    with self.assertRaises(ValueError):
                        self.generate(**values)
                    self.assertEqual(self.files(), before)
        for path in self.deps.glob("*.generated.packman.xml"):
            path.unlink()
        (self.deps / "live-services-build.json").unlink()
        before = self.files()
        with self.assertRaises(ValueError):
            self.generate(target=None)
        self.assertEqual(self.files(), before)

    def test_allowed_abi_tokens_are_preserved_without_platform_inference(self):
        self.generate(target="Explicit_Target.1-test", host="_Host_2-test.3")
        for package in self.manifest("target").findall("./dependency/package"):
            self.assertIn("-Explicit_Target.1-test-static-", package.attrib["version"])
        for package in self.manifest("host").findall("./dependency/package"):
            self.assertIn("-_Host_2-test.3-static-release", package.attrib["version"])

    def test_late_host_template_failure_does_not_publish_rendered_target(self):
        self.generate()
        host = self.deps / "live-services-host.packman.xml"
        original = host.read_text()
        for broken in (original.replace("${live_services_host_platform_abi}", "implicit-host"),
                       original.replace("</project>", "<unclosed>")):
            with self.subTest(template=broken[-30:]):
                host.write_text(broken)
                before = self.files()
                with self.assertRaises((ValueError, ET.ParseError)):
                    self.generate(target="different-explicit-target")
                self.assertEqual(self.files(), before)
        host.write_text(original)


class PlatformTableTests(unittest.TestCase):
    def test_host_names_the_platform_of_this_machine(self):
        platforms = build.platforms
        for system, machine, expected in (("Linux", "x86_64", "linux-x86_64"), ("Linux", "AMD64", "linux-x86_64"),
                                          ("Linux", "aarch64", "linux-aarch64"), ("Linux", "arm64", "linux-aarch64"),
                                          ("Windows", "AMD64", "windows-x86_64"),
                                          ("Darwin", "arm64", "macos-arm64")):
            with self.subTest(system=system, machine=machine), \
                    patch.object(platforms._host, "system", return_value=system), \
                    patch.object(platforms._host, "machine", return_value=machine):
                self.assertEqual(platforms.host().name, expected)
        for system, machine in (("Windows", "ARM64"), ("Darwin", "x86_64"), ("Linux", "riscv64")):
            with self.subTest(system=system, machine=machine), \
                    patch.object(platforms._host, "system", return_value=system), \
                    patch.object(platforms._host, "machine", return_value=machine), \
                    self.assertRaisesRegex(ValueError, "Unsupported host"):
                platforms.host()

    def test_native_artifacts_are_checked_against_the_platform_binary_format(self):
        with tempfile.TemporaryDirectory() as directory:
            pe = Path(directory) / "omni_usd_live.dll"
            header = bytearray(0x100)
            header[:2] = b"MZ"
            header[0x3C:0x40] = (0x80).to_bytes(4, "little")
            header[0x80:0x86] = b"PE\0\0" + (0x8664).to_bytes(2, "little")
            pe.write_bytes(bytes(header))
            build._native(pe, "windows-x86_64")
            with self.assertRaisesRegex(ValueError, "linux-x86_64"):
                build._native(pe, "linux-x86_64")
            header[0x84:0x86] = (0x14C).to_bytes(2, "little")
            pe.write_bytes(bytes(header))
            with self.assertRaisesRegex(ValueError, "windows-x86_64"):
                build._native(pe, "windows-x86_64")
            elf = Path(directory) / "libomni_usd_live.so"
            elf.write_bytes(b"\x7fELF\x02\x01" + b"\0" * 12 + (0x3E).to_bytes(2, "little"))
            build._native(elf, "linux-x86_64")
            with self.assertRaisesRegex(ValueError, "windows-x86_64"):
                build._native(elf, "windows-x86_64")

    def test_macos_artifacts_require_thin_arm64_macho_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "libomni_usd_live.dylib"
            header = b"\xcf\xfa\xed\xfe" + (0x0100000C).to_bytes(4, "little") + b"\0" * 24
            artifact.write_bytes(header)
            build._native(artifact, "macos-arm64")
            invalid = (header[:20], b"\xfe\xed\xfa\xcf" + header[4:],
                       header[:4] + (0x01000007).to_bytes(4, "little") + header[8:],
                       b"\xca\xfe\xba\xbe" + header[4:])
            for value in invalid:
                with self.subTest(header=value):
                    artifact.write_bytes(value)
                    with self.assertRaisesRegex(ValueError, "macos-arm64 MACHO"):
                        build._native(artifact, "macos-arm64")

    def test_macos_package_uses_dylib_and_cpython_extension(self):
        self.assertEqual(build.native_files("macos-arm64"),
                         ("lib/libomni_usd_live.dylib", "bin/live_diagnostics_sample",
                          "bin/live_participant_sample", "python/omni/usd/live/_live.cpython-312-darwin.so"))
        self.assertEqual(set(build.package_files("macos-arm64")),
                         set(build.PAYLOAD_FILES) | set(build.native_files("macos-arm64")))

    def test_each_platform_names_its_binary_format_extension_and_loader(self):
        self.assertEqual({name: (entry.binary_format, entry.machine_code, build.python_extension(name), entry.dynamic_loader)
                          for name, entry in build.platforms.PLATFORMS.items()},
                         {"linux-x86_64": ("elf", 0x3E, "python/omni/usd/live/_live.cpython-312-x86_64-linux-gnu.so",
                                           "ld-linux-x86-64.so.2"),
                          "linux-aarch64": ("elf", 0xB7, "python/omni/usd/live/_live.cpython-312-aarch64-linux-gnu.so",
                                            "ld-linux-aarch64.so.1"),
                          "macos-arm64": ("macho", 0x0100000C, "python/omni/usd/live/_live.cpython-312-darwin.so", None),
                          "windows-x86_64": ("pe", 0x8664, "python/omni/usd/live/_live.cp312-win_amd64.pyd", None)})
        # Windows places the plugin DLL beside the sample executables, where the loader finds it, and ships its
        # import library for C ABI consumers.
        windows = build.package_files("windows-x86_64")
        linux = build.package_files("linux-x86_64")
        self.assertEqual(set(windows) - set(linux), {"bin/omni_usd_live.dll", "lib/omni_usd_live.lib",
                                                     "bin/live_diagnostics_sample.exe", "bin/live_participant_sample.exe",
                                                     "python/omni/usd/live/_live.cp312-win_amd64.pyd"})
        self.assertEqual(set(linux) - set(windows), {"lib/libomni_usd_live.so", "bin/live_diagnostics_sample",
                                                     "bin/live_participant_sample",
                                                     "python/omni/usd/live/_live.cpython-312-x86_64-linux-gnu.so"})
        self.assertEqual(build.native_files("windows-x86_64"),
                         ("bin/omni_usd_live.dll", "bin/live_diagnostics_sample.exe", "bin/live_participant_sample.exe",
                          "python/omni/usd/live/_live.cp312-win_amd64.pyd"))
        self.assertEqual(build.library("windows-x86_64"), "bin/omni_usd_live.dll")
        self.assertEqual(build.library("linux-aarch64"), "lib/libomni_usd_live.so")
        for name in (None, "", "macos-x86_64", "linux-x86_64 "):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unsupported platform"):
                build.platforms.get(name)
        for name in (None, "", [], {}, "macos-x86_64", "linux-x86_64 "):
            with self.subTest(name=name):
                self.assertFalse(build.platforms.supported(name))
        self.assertTrue(all(build.platforms.supported(name) for name in build.platforms.PLATFORMS))

    def test_require_accepts_only_the_host_platform(self):
        platforms = build.platforms
        with patch.object(platforms, "host", return_value=platforms.PLATFORMS["linux-aarch64"]):
            self.assertEqual(platforms.main(["--require", "linux-aarch64"]), 0)
            with contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(platforms.main(["--require", "linux-x86_64"]), 1)
            self.assertIn("this host is linux-aarch64, not the required linux-x86_64", error.getvalue())
        with patch.object(platforms, "host", side_effect=ValueError("Unsupported host: Windows AMD64")), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(platforms.main(["--require", "linux-x86_64"]), 1)


class LiveSourceBoundaryTests(unittest.TestCase):
    def test_notices_and_sdk_identities_are_recorded_and_installed(self):
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
        self.assertIn("Copyright (C) 2011 Zilong Tan", notices)
        self.assertIn("Permission is hereby granted, free of charge", notices)
        for member in ("share/omni_usd_live/THIRD_PARTY_NOTICES.md",
                       "share/omni_usd_live/THIRD_PARTY_LICENSES.txt", "share/omni_usd_live/BOM.json",
                       "share/omni_usd_live/provenance/wire-schema/UPSTREAM-NOTICE.txt",
                       "share/omni_usd_live/provenance/flatbuffers/LICENSE.txt",
                       "share/omni_usd_live/provenance/flatbuffers/provenance.json"):
            self.assertIn(member, build.PAYLOAD_FILES)
        self.assertNotIn("share/omni_usd_live/provenance/wire-schema/LICENSE.txt", build.PAYLOAD_FILES)
        plugin = (ROOT / "cmake/LivePlugin.cmake").read_text()
        self.assertIn("RENAME UPSTREAM-NOTICE.txt", plugin)
        self.assertIn("THIRD_PARTY_NOTICES.md", plugin)
        self.assertIn('"${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_LICENSES.txt"', plugin)
        self.assertIn('"${CMAKE_CURRENT_SOURCE_DIR}/BOM.json"', plugin)
        identities = (ROOT / "deps/README.md").read_text().split("## Host SDK identities", 1)[1]
        for identity in ("usd.py312.manylinux_2_35_x86_64.ov-full-monolithic.release", "0.25.11.kit.5-gl.21209",
                         "3.12.13+nv3-manylinux_2_35_x86_64", "2.11.1", "Carbonite and Omnitrace are not build inputs"):
            self.assertIn(identity, identities)

    def test_gcc_keeps_the_include_directory_spelling_of_system_headers(self):
        # A shorter canonical path would name the package manager's cache, which no -ffile-prefix-map covers.
        self.assertIn('add_compile_options("-fno-canonical-system-headers")', (ROOT / "cmake/LiveBuildPaths.cmake").read_text())

    def test_retired_live_trees_and_client_update_hooks_are_absent(self):
        for name in ("source/live/library", "source/live/include", "source/live/tests",
                     "source/live/samples/DancingCubes", "source/live/samples/Python",
                     "source/live/samples/SiemensDemo", "source/live/samples/common",
                     "omni-client-library.txt", "tools/ci/update-client-library"):
            path = ROOT / name
            with self.subTest(path=name):
                self.assertFalse(path.is_file() or (path.is_dir() and any(path.rglob("*"))))

    def test_independent_codec_type_table_and_public_headers_remain(self):
        for name in ("source/live/wire/UsdValueTypes.inc", "include/omni/usd/live/Live.h",
                     "include/omni/usd/live/LiveUsd.h"):
            with self.subTest(path=name):
                self.assertTrue((ROOT / name).is_file())
        self.assertIn("USD_TYPE", (ROOT / "source/live/wire/UsdValueTypes.inc").read_text())

    def test_active_build_configuration_has_no_omni_client_hooks(self):
        files = ("CMakeLists.txt", "tools/live/build.py", "deps/live-services-target.packman.xml",
                 "deps/live-services-host.packman.xml")
        forbidden = re.compile(r"use_client_library|omniClientDir|client-library-deps|omni_client_library|omni-client-library\.txt|library/v[12]")
        for name in files:
            with self.subTest(path=name):
                self.assertIsNone(forbidden.search((ROOT / name).read_text()))

    def test_packaged_metadata_uses_one_canonical_production_library(self):
        for platform in build.platforms.PLATFORMS:
            metadata = json.loads(build.plug_info(ROOT, platform))
            self.assertEqual(len(metadata["Plugins"]), 1)
            plugin = metadata["Plugins"][0]
            self.assertEqual((plugin["Name"], plugin["Root"], plugin["LibraryPath"], plugin["ResourcePath"]),
                             ("omni_usd_live", "..", build.library(platform), "resources"))
        types = plugin["Info"]["Types"]
        self.assertEqual(set(types), {"OmniUsdLiveResolver", "OmniUsdLiveNativeFileFormat"})
        self.assertEqual(types["OmniUsdLiveResolver"]["uriSchemes"], ["live"])
        self.assertEqual(types["OmniUsdLiveNativeFileFormat"]["extensions"], ["live"])


class LivePackageStageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # A synthetic repository with root license and client notices.
        self.root = Path(temporary.name) / "client"
        self.stage = self.root / "_build/linux-x86_64/release/usd/omniverse/live"
        self.stage.mkdir(parents=True)
        self.inputs = {}
        for name, source in build.NOTICES.items():
            path = self.root / source
            path.write_text("independent synthetic source " + name)
            self.inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        source_names = ("CMakeLists.txt", "tools/live/build.py",
                        "source/live/runtime/Fixture.h", "include/omni/usd/live/Live.h",
                        "source/live/samples/diagnostics/main.cpp",
                        "source/live/samples/diagnostics/DiagnosticLog.cpp",
                        "source/live/samples/diagnostics/DiagnosticLog.h",
                        "source/bindings/python/omni/usd/live/Native.cpp",
                        "source/bindings/python/omni/usd/live/__init__.py",
                        "source/bindings/python/omni/usd/live/_host.py",
                        "source/bindings/python/omni/usd/live/__init__.pyi",
                        "source/bindings/python/omni/usd/live/py.typed")
        for name in source_names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("independent synthetic source " + name)
            self.inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        root_patch = patch.object(build, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        elf = bytearray(20)
        elf[:6], elf[18:20] = b"\x7fELF\x02\x01", b"\x3e\x00"
        self.members = {
            "lib/libomni_usd_live.so": bytes(elf) + b"synthetic plugin, never executed",
            "python/omni/usd/live/_live.cpython-312-x86_64-linux-gnu.so": bytes(elf) + b"synthetic Python adapter, never executed",
            "python/omni/usd/live/__init__.py": b"synthetic Python wrapper",
            "python/omni/usd/live/_host.py": b"synthetic Python host wrapper",
            "python/omni/usd/live/_kit_host.py": b"synthetic Kit lifecycle owner",
            "python/omni/usd/live/_kit_sessions.py": b"synthetic Kit session catalogue",
            "python/omni/usd/live/_kit_root.py": b"synthetic Kit root policy",
            "python/omni/kit/live_services/__init__.py": b"synthetic Kit entry point",
            "python/omni/kit/live_services/_bootstrap.py": b"synthetic Kit preflight",
            "config/extension.toml": b"synthetic Kit extension metadata",
            "config/live-sdk.json": b'{"schema_version":1,"usd_profile":"stock-25.11"}',
            "python/omni/usd/live/sample.py": b"synthetic Python workflow sample",
            "python/omni/usd/live/__init__.pyi": b"synthetic Python typing",
            "python/omni/usd/live/py.typed": b"",
            "bin/live_participant_sample": bytes(elf) + b"synthetic participant sample, never executed",
            "bin/live_diagnostics_sample": bytes(elf) + b"synthetic diagnostics sample, never executed",
            "include/omni/usd/live/Live.h": b"synthetic public C header",
            "include/omni/usd/live/LiveUsd.h": b"synthetic public USD header",
            "resources/plugInfo.json": build.plug_info(ROOT, "linux-x86_64"),
            "share/omni_usd_live/LICENSE": b"synthetic Apache license",
            "share/omni_usd_live/NOTICE": b"synthetic NVIDIA notice",
            "share/omni_usd_live/THIRD_PARTY_NOTICES.md": b"synthetic third-party notices",
            "share/omni_usd_live/THIRD_PARTY_LICENSES.txt": b"synthetic third-party license texts",
            "share/omni_usd_live/BOM.json": b'{"synthetic bill of materials": true}',
            "share/omni_usd_live/provenance/live-proto.md": b"synthetic protocol provenance",
            "share/omni_usd_live/provenance/wire-schema/provenance.json": b'{"synthetic": true}',
            "share/omni_usd_live/provenance/wire-schema/UPSTREAM-NOTICE.txt": b"synthetic schema license",
            "share/omni_usd_live/provenance/flatbuffers/LICENSE.txt": b"synthetic FlatBuffers license",
            "share/omni_usd_live/provenance/flatbuffers/provenance.json": b'{"synthetic flatbuffers": true}',
        }
        for name, value in self.members.items():
            path = self.stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            if name in {"bin/live_diagnostics_sample", "bin/live_participant_sample"}:
                path.chmod(0o755)
        self.record = {"schema_version": 1, "delegate_version": 2, "owner": "native-live-build",
                       "platform": "linux-x86_64", "configuration": "release",
                       "inputs": self.inputs,
                       "files": {name: hashlib.sha256(value).hexdigest() for name, value in self.members.items()}}
        support = {"platform": "linux-x86_64", "config": "release", "tools": {}, "sdks": {}}
        identity = {"tools": {}, "trees": {}}
        for name in ("cmake", "ninja", "cc", "cxx", "protoc", "grpc_cpp_plugin", "python"):
            tool = (self.root / "support/python_root/bin/python3.12" if name == "python"
                    else self.root / "support/tools" / name)
            tool.parent.mkdir(parents=True, exist_ok=True)
            content = ("synthetic executable " + name).encode()
            tool.write_bytes(content)
            tool.chmod(0o755)
            support["tools"][name] = str(tool)
            identity["tools"][name] = {"path": str(tool), "sha256": hashlib.sha256(content).hexdigest()}
        for name in ("usd_root", "python_root", "pybind11_root", "dependency_root"):
            base = self.root / "support" / name
            support["sdks"][name] = str(base)
            parts = ["include", "lib"] if name != "dependency_root" else [f"{package}-release/{part}" for package in PINS for part in ("include", "lib")]
            if name == "pybind11_root":
                parts = ["include"]
            for part in parts:
                directory = base / part
                directory.mkdir(parents=True)
                content = (name + "/" + part).encode()
                (directory / "fixture").write_bytes(content)
                # Independent one-file golden tree: relative name, NUL, digest.
                tree_hash = hashlib.sha256(b"fixture\0" + hashlib.sha256(content).digest()).hexdigest()
                identity["trees"][name + "/" + part] = {"path": str(directory), "sha256": tree_hash}
        support["usd_profile"] = identity["usd_profile"] = "stock-25.11"
        self.record["support_inputs"], self.record["support_identity"] = support, identity
        self.write_manifest()

    def write_manifest(self):
        (self.stage / "package-manifest.json").write_text(json.dumps(self.record, sort_keys=True))

    def test_complete_stage_is_valid_without_running_artifacts(self):
        with patch("subprocess.run", side_effect=AssertionError("stage validation ran an artifact")):
            self.assertEqual(build.validate_stage(self.stage), self.record)

    def test_missing_modified_and_extra_package_members_reject(self):
        for name, original in self.members.items():
            path = self.stage / name
            with self.subTest(member=name, mutation="missing"):
                path.unlink()
                with self.assertRaises((ValueError, OSError)):
                    build.validate_stage(self.stage)
                path.write_bytes(original)
                if name in {"bin/live_diagnostics_sample", "bin/live_participant_sample"}:
                    path.chmod(0o755)
            with self.subTest(member=name, mutation="changed"):
                path.write_bytes(original + b" ")
                with self.assertRaises((ValueError, OSError)):
                    build.validate_stage(self.stage)
                path.write_bytes(original)
            self.assertEqual(build.validate_stage(self.stage), self.record)
        extra = self.stage / "lib/extra-client.so"
        extra.write_bytes(b"must not merge stale package contents")
        with self.assertRaises(ValueError):
            build.validate_stage(self.stage)

    def test_extra_python_code_bytecode_or_extension_is_rejected_even_when_manifest_claims_it(self):
        for name in ("python/omni/usd/live/extra.py", "python/omni/usd/live/__pycache__/__init__.cpython-312.pyc",
                     "python/omni/usd/live/__init__.pyo", "python/omni/usd/live/_live.cpython-311-x86_64-linux-gnu.so"):
            with self.subTest(member=name):
                path = self.stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"unreviewed Python package content")
                self.record["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.write_manifest()
                with self.assertRaises(ValueError):
                    build.validate_stage(self.stage)
                path.unlink()
                del self.record["files"][name]
                self.write_manifest()
                self.assertEqual(build.validate_stage(self.stage), self.record)

    def test_empty_python_marker_is_valid_but_empty_module_or_binary_is_not(self):
        self.assertEqual((self.stage / "python/omni/usd/live/py.typed").read_bytes(), b"")
        for name in ("python/omni/usd/live/__init__.py", "python/omni/usd/live/_host.py", "python/omni/usd/live/__init__.pyi",
                     "python/omni/usd/live/_live.cpython-312-x86_64-linux-gnu.so"):
            with self.subTest(member=name):
                path = self.stage / name
                original = path.read_bytes()
                path.write_bytes(b"")
                self.record["files"][name] = hashlib.sha256(b"").hexdigest()
                self.write_manifest()
                with self.assertRaises(ValueError):
                    build.validate_stage(self.stage)
                path.write_bytes(original)
                self.record["files"][name] = hashlib.sha256(original).hexdigest()
                self.write_manifest()
                self.assertEqual(build.validate_stage(self.stage), self.record)

    def test_python_source_sdk_and_interpreter_changes_make_stage_stale(self):
        support = self.record["support_inputs"]
        paths = [self.root / "source/bindings/python/omni/usd/live" / name
                 for name in ("Native.cpp", "__init__.py", "__init__.pyi", "py.typed")]
        paths += [Path(support["sdks"]["pybind11_root"]) / "include/fixture",
                  Path(support["sdks"]["python_root"]) / "include/fixture", Path(support["tools"]["python"])]
        for path in paths:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b" changed in place")
                with self.assertRaises(ValueError):
                    build.validate_stage(self.stage)
                path.write_bytes(original)
                self.assertEqual(build.validate_stage(self.stage), self.record)

    def test_source_mutations_and_new_source_files_make_stage_stale(self):
        source = self.root / "source/live/runtime/Fixture.h"
        original = source.read_bytes()
        source.write_bytes(original + b" changed")
        with self.assertRaises(ValueError):
            build.validate_stage(self.stage)
        source.write_bytes(original)
        (source.parent / "Added.h").write_text("new source omitted from previous provenance")
        with self.assertRaises(ValueError):
            build.validate_stage(self.stage)

    def test_notice_source_changes_make_stage_stale(self):
        sources = {"LICENSE": self.root.parent / "LICENSE", "NOTICE": self.root / "NOTICE",
                   "THIRD_PARTY_NOTICES.md": self.root / "THIRD_PARTY_NOTICES.md",
                   "THIRD_PARTY_LICENSES.txt": self.root / "THIRD_PARTY_LICENSES.txt",
                   "BOM.json": self.root / "BOM.json"}
        # The root index describes both parts and is not the client package's notice payload.
        (self.root.parent / "THIRD_PARTY_NOTICES.md").write_text("repository notice index")
        (self.root.parent / "NOTICE").write_text("repository and service attribution")
        self.assertEqual(build.validate_stage(self.stage), self.record)
        for name, path in sources.items():
            with self.subTest(notice=name):
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                with self.assertRaisesRegex(ValueError, "[Ss]ource"):
                    build.validate_stage(self.stage)
                path.write_bytes(original)
                self.assertEqual(build.validate_stage(self.stage), self.record)

    def test_nonexecutable_tools_are_not_a_complete_package(self):
        for name in ("live_diagnostics_sample", "live_participant_sample"):
            with self.subTest(tool=name):
                with not_executable(self.stage / "bin" / name), self.assertRaises(ValueError):
                    build.validate_stage(self.stage)
                self.assertEqual(build.validate_stage(self.stage), self.record)

    def test_diagnostics_sample_requires_elf_even_with_claimed_hash(self):
        name = "bin/live_diagnostics_sample"
        path = self.stage / name
        path.write_bytes(b"not an executable ELF file")
        self.record["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.stage / build.MANIFEST).write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError, "ELF"):
            build.validate_stage(self.stage)

    def test_redirected_metadata_rejects_even_with_updated_member_hash(self):
        path = self.stage / "resources/plugInfo.json"
        original = json.loads(path.read_text())
        for key, value in (("Root", "../../outside"), ("LibraryPath", "/outside/libomni_usd_live.so")):
            with self.subTest(key=key):
                metadata = json.loads(json.dumps(original))
                metadata["Plugins"][0][key] = value
                path.write_text(json.dumps(metadata))
                self.record["files"]["resources/plugInfo.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.write_manifest()
                with self.assertRaises(ValueError):
                    build.validate_stage(self.stage)

    def test_symlink_members_and_unrecognized_stage_owner_reject(self):
        path = self.stage / "include/omni/usd/live/Live.h"
        external = self.root / "outside-header"
        external.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(external)
        with self.assertRaises(ValueError):
            build.validate_stage(self.stage)
        path.unlink()
        path.write_bytes(external.read_bytes())
        self.record["owner"] = "other-builder"
        self.write_manifest()
        with self.assertRaises(ValueError):
            build.validate_stage(self.stage)

    def test_sibling_resolver_is_rejected_without_changing_either_package(self):
        sibling = self.stage.parent / "resolver/resources/plugInfo.json"
        sibling.parent.mkdir(parents=True)
        sibling.write_text('{"Plugins": [{"Name": "sibling-resolver"}]}')
        before = {path.relative_to(self.stage): path.read_bytes() for path in self.stage.rglob("*") if path.is_file()}
        with self.assertRaises(ValueError):
            build.validate_stage(self.stage)
        self.assertEqual({path.relative_to(self.stage): path.read_bytes() for path in self.stage.rglob("*") if path.is_file()}, before)
        self.assertEqual(sibling.read_text(), '{"Plugins": [{"Name": "sibling-resolver"}]}')

    def test_aarch64_stage_requires_aarch64_elf_members(self):
        x86_extension = "python/omni/usd/live/_live.cpython-312-x86_64-linux-gnu.so"
        extension = "python/omni/usd/live/_live.cpython-312-aarch64-linux-gnu.so"
        (self.stage / x86_extension).rename(self.stage / extension)
        self.record["files"][extension] = self.record["files"].pop(x86_extension)
        self.record["platform"] = self.record["support_inputs"]["platform"] = "linux-aarch64"
        native = ("lib/libomni_usd_live.so", "bin/live_diagnostics_sample", "bin/live_participant_sample", extension)

        def machine(value):
            for name in native:
                path = self.stage / name
                path.write_bytes(path.read_bytes()[:18] + value + path.read_bytes()[20:])
                self.record["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.write_manifest()

        machine(b"\xb7\x00")
        self.assertEqual(build.validate_stage(self.stage, expected_platform="linux-aarch64"), self.record)
        machine(b"\x3e\x00")
        with self.assertRaisesRegex(ValueError, "linux-aarch64 ELF"):
            build.validate_stage(self.stage)

    def test_macos_stage_checks_native_members_and_sample_execute_permission(self):
        for source, destination in zip(build.native_files("linux-x86_64"), build.native_files("macos-arm64")):
            (self.stage / source).rename(self.stage / destination)
            (self.stage / destination).write_bytes(b"\xcf\xfa\xed\xfe" + (0x0100000C).to_bytes(4, "little") + b"\0" * 24)
        (self.stage / "resources/plugInfo.json").write_bytes(build.plug_info(ROOT, "macos-arm64"))
        self.record["platform"] = self.record["support_inputs"]["platform"] = "macos-arm64"
        self.record["files"] = {name: build._hash(self.stage / name) for name in build.package_files("macos-arm64")}
        self.write_manifest()
        self.assertEqual(build.validate_stage(self.stage, expected_platform="macos-arm64"), self.record)
        with not_executable(self.stage / "bin/live_participant_sample"):
            with self.assertRaisesRegex(ValueError, "is not executable"):
                build.validate_stage(self.stage)
        (self.stage / build.library("macos-arm64")).write_bytes(b"\x7fELF\x02\x01" + b"\0" * 58)
        with self.assertRaisesRegex(ValueError, "macos-arm64 MACHO"):
            build.validate_stage(self.stage)

    def test_requested_platform_and_configuration_must_match_stage(self):
        self.assertEqual(build.validate_stage(self.stage, expected_platform="linux-x86_64", expected_configuration="release"), self.record)
        for kwargs in ({"expected_platform": "linux-aarch64"}, {"expected_configuration": "debug"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build.validate_stage(self.stage, **kwargs)
        for field, wrong in (("platform", "windows-x86_64"), ("configuration", "relwithdebinfo")):
            original = self.record[field]
            self.record[field] = wrong
            self.write_manifest()
            with self.assertRaises(ValueError):
                build.validate_stage(self.stage)
            self.record[field] = original
        self.write_manifest()
        self.assertEqual(build.validate_stage(self.stage), self.record)

    def test_same_path_sdk_and_tool_changes_invalidate_stage(self):
        support = self.record["support_inputs"]
        paths = [Path(support["tools"]["protoc"]), Path(support["sdks"]["usd_root"]) / "include/fixture",
                 Path(support["sdks"]["dependency_root"]) / "grpc-release/lib/fixture"]
        for path in paths:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b" changed in place")
                with self.assertRaises(ValueError):
                    build.validate_stage(self.stage)
                path.write_bytes(original)
                self.assertEqual(build.validate_stage(self.stage), self.record)

    def test_retargeted_sdk_and_tool_links_cannot_reuse_previous_identity(self):
        support = self.record["support_inputs"]
        for path in (Path(support["tools"]["grpc_cpp_plugin"]), Path(support["sdks"]["usd_root"]) / "lib/fixture"):
            with self.subTest(path=path):
                original, mode = path.read_bytes(), path.stat().st_mode
                first = self.root / (path.parent.name + "-first-target")
                second = self.root / (path.parent.name + "-second-target")
                first.write_bytes(original); first.chmod(mode)
                second.write_bytes(b"different binary at new target"); second.chmod(mode)
                path.unlink(); path.symlink_to(first)
                self.assertEqual(build.validate_stage(self.stage), self.record)
                path.unlink(); path.symlink_to(second)
                with self.assertRaises(ValueError):
                    build.validate_stage(self.stage)
                path.unlink(); path.write_bytes(original); path.chmod(mode)

    def test_execute_overrides_python_environment_without_mutating_sdk_identity(self):
        support = self.record["support_inputs"]
        cache = self.root / "synthetic-build-cache"
        plan = {"root": str(self.root), "platform": "linux-x86_64", "config": "release",
                "build_dir": str(cache), "target": str(self.stage), "tools": support["tools"], "sdks": support["sdks"],
                "commands": [["synthetic-cmake", "configure"], ["synthetic-cmake", "build"],
                             ["synthetic-cmake", "--install", str(cache), "--prefix", "placeholder"]]}
        bytecode = Path(support["sdks"]["python_root"]) / "lib/__pycache__/sysconfig.cpython-312.pyc"
        def interpreter_effect(environment):
            if environment.get("PYTHONDONTWRITEBYTECODE") != "1":
                bytecode.parent.mkdir(exist_ok=True)
                bytecode.write_bytes(b"synthetic sysconfig import bytecode")
        # Interpreter caches do not change SDK identity; execute also prevents
        # its children from writing them, even with an inherited 0.
        interpreter_effect({"PYTHONDONTWRITEBYTECODE": "0"})
        self.assertEqual(build.validate_stage(self.stage), self.record)
        bytecode.unlink()
        bytecode.parent.rmdir()
        self.assertEqual(build.validate_stage(self.stage), self.record)
        original_identity = self.record["support_identity"]
        calls = []
        def run(command, **kwargs):
            environment = kwargs.get("env", {})
            interpreter_effect(environment)
            self.assertEqual(environment.get("PYTHONDONTWRITEBYTECODE"), "1")
            self.assertEqual(environment.get("PYTHONNOUSERSITE"), "1")
            self.assertNotIn("PYTHONHOME", environment)
            self.assertNotIn("PYTHONPATH", environment)
            self.assertEqual(environment.get("LIVE_TEST_ENV_WITNESS"), "preserved")
            self.assertEqual(kwargs["cwd"], self.root)
            self.assertTrue(kwargs["check"])
            calls.append(list(command))
            if "--install" not in command:
                return
            prefix = Path(command[command.index("--prefix") + 1])
            for name, data in self.members.items():
                destination = prefix / ("lib/omni_usd_live/" + name if name.startswith(("lib/", "resources/", "python/", "config/")) else name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                if name in {"bin/live_diagnostics_sample", "bin/live_participant_sample"}:
                    destination.chmod(0o755)
        with patch.dict(os.environ, {"PYTHONDONTWRITEBYTECODE": "0", "PYTHONNOUSERSITE": "0",
                                     "LIVE_TEST_ENV_WITNESS": "preserved", "PYTHONHOME": "/hostile/python-home",
                                     "PYTHONPATH": "/hostile/python-modules"}), patch("subprocess.run", side_effect=run):
            build.execute(plan)
            self.assertEqual(os.environ["PYTHONDONTWRITEBYTECODE"], "0")
            self.assertEqual(os.environ["PYTHONNOUSERSITE"], "0")
            self.assertEqual(os.environ["PYTHONHOME"], "/hostile/python-home")
            self.assertEqual(os.environ["PYTHONPATH"], "/hostile/python-modules")
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[:2], plan["commands"][:2])
        self.assertIn("--install", calls[2])
        self.assertFalse(bytecode.exists())
        self.assertFalse(list(self.stage.parent.glob(".live-stage-*")))
        staged = build.validate_stage(self.stage)
        self.assertEqual(staged["support_identity"], original_identity)
        self.assertEqual(staged["files"], self.record["files"])
        self.assertEqual(staged["inputs"], self.inputs)

    def test_failed_install_and_failed_publication_preserve_previous_complete_stage(self):
        support = self.record["support_inputs"]
        cache = self.root / "synthetic-build-cache"
        plan = {"root": str(self.root), "platform": "linux-x86_64", "config": "release",
                "build_dir": str(cache), "target": str(self.stage), "tools": support["tools"], "sdks": support["sdks"],
                "commands": [["synthetic-cmake", "configure"], ["synthetic-cmake", "build"],
                             ["synthetic-cmake", "--install", str(cache), "--prefix", "placeholder"]]}
        before = {p.relative_to(self.stage): p.read_bytes() for p in self.stage.rglob("*") if p.is_file()}
        original_rename = Path.rename
        for mode in ("install-failure", "publish-failure"):
            calls, moves = [], []
            def run(command, **kwargs):
                calls.append(command)
                if "--install" not in command:
                    return
                if mode == "install-failure":
                    raise subprocess.CalledProcessError(31, command)
                prefix = Path(command[command.index("--prefix") + 1])
                for name, data in self.members.items():
                    destination = prefix / ("lib/omni_usd_live/" + name if name.startswith(("lib/", "resources/", "python/", "config/")) else name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    if name in {"bin/live_diagnostics_sample", "bin/live_participant_sample"}:
                        destination.chmod(0o755)
            def rename(path, destination):
                moves.append((path, Path(destination)))
                if path.name == "package" and Path(destination) == self.stage:
                    raise OSError("synthetic final package rename failure")
                return original_rename(path, destination)
            with self.subTest(mode=mode), patch("subprocess.run", side_effect=run), patch.object(Path, "rename", rename):
                with self.assertRaises((subprocess.CalledProcessError, OSError)):
                    build.execute(plan)
            self.assertEqual(len(calls), 3)
            self.assertEqual({p.relative_to(self.stage): p.read_bytes() for p in self.stage.rglob("*") if p.is_file()}, before)
            self.assertFalse(list(self.stage.parent.glob(".live-stage-*")))
            self.assertEqual(build.validate_stage(self.stage), self.record)
            if mode == "publish-failure":
                self.assertTrue(any(source == self.stage for source, destination in moves), "old package was not moved before failure")
                self.assertTrue(any(source.name == "previous" and destination == self.stage for source, destination in moves),
                                "old package restoration was not exercised")


class LiveBuildPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # A synthetic repository with root license and client notices.
        self.root = Path(temporary.name) / "repo with spaces" / "client"
        self.root.mkdir(parents=True)
        for source in build.NOTICES.values():
            (self.root / source).write_text("synthetic build input")
        for name in ("CMakeLists.txt", "tools/live/build.py"):
            self.write(name, "synthetic build input")
        args = {"root": str(self.root), "platform": "linux-x86_64", "config": "release",
                "target_dir": str(self.root / "output"), "build_dir": str(self.root / "cache"), "dry_run": True}
        for key in ("cmake", "ninja", "cc", "cxx", "protoc", "grpc_cpp_plugin"):
            tool = self.write("host tools/" + key, "synthetic executable, never run")
            tool.chmod(0o755)
            args[key] = str(tool)
        for key in ("usd_root", "python_root", "pybind11_root", "dependency_root"):
            directory = self.root / "sdks" / key
            (directory / "include").mkdir(parents=True)
            args[key] = str(directory)
        self.args = SimpleNamespace(**args)
        self.write("sdks/usd_root/include/pxr/pxr.h", "#define PXR_VERSION 2511\n#define PXR_NS pxr\n"
                   "#define PXR_INTERNAL_NS pxrInternal_v0_25_11__pxrReserved__\n#define PXR_PYTHON_SUPPORT_ENABLED\n")
        self.write("sdks/python_root/include/python3.12/patchlevel.h", "#define PY_MAJOR_VERSION 3\n#define PY_MINOR_VERSION 12\n")
        for library in ("usd_sdf", "usd_ar", "usd_plug", "usd_usdGeom", "usd_vt", "usd_gf", "usd_tf",
                        "usd_arch", "usd_trace", "usd_work", "usd_python", "usd_js", "tbb"):
            self.write("sdks/usd_root/lib/lib" + library + ".so", "synthetic SDK library")
        self.write("sdks/python_root/lib/libpython3.12.so", "synthetic Python library")
        self.write("sdks/python_root/bin/python3.12", "synthetic selected Python interpreter, never run").chmod(0o755)
        self.write("sdks/pybind11_root/include/pybind11/pybind11.h", "synthetic pybind11 header")
        self.write("sdks/pybind11_root/include/pybind11/detail/common.h",
                   "#define PYBIND11_VERSION_MAJOR 2\n#define PYBIND11_VERSION_MINOR 11\n#define PYBIND11_VERSION_PATCH 1\n")
        for config in ("debug", "release"):
            for package in PINS:
                for directory in ("include", "lib"):
                    (Path(self.args.dependency_root) / f"{package}-{config}" / directory).mkdir(parents=True)
        host_patch = patch.object(build.platforms, "host", return_value=build.platforms.PLATFORMS["linux-x86_64"])
        host_patch.start()
        self.addCleanup(host_patch.stop)

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def snapshot(self, root=None):
        root = root or self.root
        return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mode)
                for p in root.rglob("*") if p.is_file()}

    def test_debug_and_release_plans_use_host_tools_and_one_production_plugin(self):
        before = self.snapshot()
        with patch("subprocess.run", side_effect=AssertionError("plan invoked a tool")):
            for config in ("debug", "release"):
                self.args.config = config
                plan = build.plan(self.args)
                settings = dict(arg[2:].split("=", 1) for arg in plan["commands"][0] if arg.startswith("-D"))
                self.assertEqual(settings["CMAKE_BUILD_TYPE"], {"debug": "Debug", "release": "Release"}[config])
                self.assertEqual(settings["LIVE_DEPENDENCY_CONFIG"], config)
                self.assertEqual(settings["LIVE_PROTOC"], self.args.protoc)
                self.assertEqual(settings["LIVE_GRPC_CPP_PLUGIN"], self.args.grpc_cpp_plugin)
                self.assertFalse(settings["LIVE_PROTOC"].startswith(self.args.dependency_root))
                self.assertEqual(settings["CMAKE_C_COMPILER"], self.args.cc)
                self.assertEqual(settings["CMAKE_CXX_COMPILER"], self.args.cxx)
                self.assertEqual(settings["LIVE_USD_PYTHON_ROOT"], self.args.python_root)
                self.assertEqual(settings["BUILD_TESTING"], "OFF")
                self.assertEqual(settings["CMAKE_BUILD_WITH_INSTALL_RPATH"], "ON")
                self.assertEqual(settings["LIVE_BUILD_PLUGIN"], "ON")
                self.assertEqual(settings["LIVE_BUILD_PYTHON"], "ON")
                self.assertEqual(settings["LIVE_PYBIND11_ROOT"], self.args.pybind11_root)
                interpreter = str(Path(self.args.python_root) / "bin/python3.12")
                self.assertEqual(settings["LIVE_PYTHON_EXECUTABLE"], interpreter)
                self.assertEqual(plan["tools"]["python"], interpreter)
                self.assertEqual(settings["LIVE_BUILD_RESOLVER_RUNTIME"], "OFF")
                self.assertEqual(settings["CMAKE_INSTALL_DATADIR"], "share")
                targets = plan["commands"][1]
                self.assertEqual(targets[targets.index("--parallel") + 1], "2")
                self.assertEqual(targets[targets.index("--target") + 1:], ["omni_usd_live_python", "live_diagnostics_sample", "live_participant_sample"])
                self.assertIn(str(self.root.resolve()), plan["commands"][0])  # Spaces remain one argv element.
                self.assertEqual(plan["target"], str(Path(self.args.target_dir) / "usd/omniverse/live"))
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(Path(self.args.build_dir).exists())
        self.assertFalse(Path(self.args.target_dir).exists())

    def test_kit_profile_requires_matching_namespace_and_monolithic_library(self):
        self.args.usd_profile = 'kit-25.11'
        with self.assertRaisesRegex(ValueError, 'USD header does not match'):
            build.plan(self.args)
        header = self.root / 'sdks/usd_root/include/pxr/pxr.h'
        header.write_text(header.read_text().replace('pxrInternal_', 'ovInternal_').replace('PXR_NS pxr', 'PXR_NS ov'))
        with self.assertRaises(FileNotFoundError):
            build.plan(self.args)
        self.write('sdks/usd_root/lib/libov_25.11usd_ms.so', 'synthetic Kit monolithic SDK')
        # A Kit plan must not require or fall back to the split stock libraries.
        for path in (self.root / 'sdks/usd_root/lib').glob('libusd_*.so'):
            path.unlink()
        plan = build.plan(self.args)
        self.assertEqual(plan['usd_profile'], 'kit-25.11')
        self.assertIn('-DLIVE_USD_PROFILE=kit-25.11', plan['commands'][0])
        self.assertEqual(build._support_identity(plan)['usd_profile'], 'kit-25.11')
        self.args.usd_profile = 'stock-25.11'
        with self.assertRaisesRegex(ValueError, 'USD header does not match'):
            build.plan(self.args)
        self.args.usd_profile = 'unrecognized'
        with self.assertRaisesRegex(ValueError, 'Unsupported USD profile'):
            build.plan(self.args)

    def test_python_profile_requires_pinned_headers_and_executable_before_build(self):
        cases = (("sdks/pybind11_root/include/pybind11/detail/common.h", "#define PYBIND11_VERSION_MAJOR 2\n#define PYBIND11_VERSION_MINOR 12\n#define PYBIND11_VERSION_PATCH 1\n"),
                 ("sdks/pybind11_root/include/pybind11/pybind11.h", None),
                 ("sdks/python_root/bin/python3.12", None))
        for relative, replacement in cases:
            path = self.root / relative
            original, mode = path.read_bytes(), path.stat().st_mode
            with self.subTest(path=relative):
                if replacement is None:
                    path.unlink()
                else:
                    path.write_text(replacement)
                before = self.snapshot()
                with patch("subprocess.run", side_effect=AssertionError("invalid Python profile ran tool")):
                    with self.assertRaises((ValueError, OSError)):
                        build.plan(self.args)
                self.assertEqual(self.snapshot(), before)
                path.write_bytes(original)
                path.chmod(mode)
        with not_executable(self.root / "sdks/python_root/bin/python3.12"), self.assertRaises(ValueError):
            build.plan(self.args)
        self.assertFalse(Path(self.args.build_dir).exists())

    def test_make_selection_and_bounded_parallelism_are_explicit(self):
        make = self.write("host tools/make", "synthetic make, never run")
        make.chmod(0o755)
        self.args.make, self.args.ninja, self.args.parallel = str(make), None, 3
        before = self.snapshot()
        with patch("subprocess.run", side_effect=AssertionError("make plan ran a tool")):
            plan = build.plan(self.args)
        command = plan["commands"][0]
        self.assertEqual(command[command.index("-G") + 1], "Unix Makefiles")
        self.assertIn("-DCMAKE_MAKE_PROGRAM=" + str(make), command)
        self.assertNotIn("ninja", plan["tools"])
        self.assertEqual(plan["commands"][1][plan["commands"][1].index("--parallel") + 1], "3")
        self.assertEqual(self.snapshot(), before)
        for parallel in (0, -1, 65, "2"):
            self.args.parallel = parallel
            with self.subTest(parallel=parallel), self.assertRaises(ValueError):
                build.plan(self.args)
        self.args.parallel = 2
        self.args.ninja = str(make)
        with self.assertRaises(ValueError):
            build.plan(self.args)
        self.args.ninja = self.args.make = None
        with self.assertRaises(ValueError):
            build.plan(self.args)
        self.assertFalse(Path(self.args.build_dir).exists())

    def test_plan_builds_only_for_the_host_platform(self):
        with patch.object(build.platforms, "host", return_value=build.platforms.PLATFORMS["linux-aarch64"]):
            with self.assertRaisesRegex(ValueError, "does not match this host"):
                build.plan(self.args)
            self.args.platform = "linux-aarch64"
            self.assertEqual(build.plan(self.args)["platform"], "linux-aarch64")
        with self.assertRaisesRegex(ValueError, "does not match this host"):
            build.plan(self.args)
        # A Windows SDK keeps import libraries in lib, Python headers in include and the interpreter at its root.
        self.args.platform = "windows-x86_64"
        with patch.object(build.platforms, "host", return_value=build.platforms.PLATFORMS["windows-x86_64"]):
            with self.assertRaisesRegex(FileNotFoundError, r"usd_sdf\.lib"):
                build.plan(self.args)
            for library in ("usd_sdf", "usd_ar", "usd_plug", "usd_usdGeom", "usd_vt", "usd_gf", "usd_tf",
                            "usd_arch", "usd_trace", "usd_work", "usd_python", "usd_js", "tbb"):
                self.write("sdks/usd_root/lib/" + library + ".lib", "synthetic import library")
            self.write("sdks/python_root/include/patchlevel.h", "#define PY_MAJOR_VERSION 3\n#define PY_MINOR_VERSION 12\n")
            self.write("sdks/python_root/libs/python312.lib", "synthetic Python import library")
            self.write("sdks/python_root/python.exe", "synthetic Windows interpreter, never run").chmod(0o755)
            self.assertEqual(build.plan(self.args)["platform"], "windows-x86_64")
        self.args.platform = "macos-x86_64"
        with self.assertRaisesRegex(ValueError, "Unsupported platform"):
            build.plan(self.args)

    def test_macos_plan_requires_dylib_sdk_and_preserves_identity_validation(self):
        self.args.platform = "macos-arm64"
        with patch.object(build.platforms, "host", return_value=build.platforms.get("macos-arm64")):
            with self.assertRaisesRegex(FileNotFoundError, r"libusd_sdf\.dylib"):
                build.plan(self.args)
            for library in (Path(self.args.usd_root) / "lib").glob("*.so"):
                library.rename(library.with_suffix(".dylib"))
            with self.assertRaisesRegex(FileNotFoundError, r"libpython3.12\.dylib"):
                build.plan(self.args)
            self.write("sdks/python_root/lib/libpython3.12.dylib", "synthetic Python dylib")
            before = self.snapshot()
            result = build.plan(self.args)
            self.assertEqual(result["platform"], "macos-arm64")
            self.assertEqual(result["usd_profile"], "stock-25.11")
            self.assertEqual(result["tools"]["python"], str(Path(self.args.python_root) / "bin/python3.12"))
            self.assertEqual(self.snapshot(), before)
            self.write("sdks/python_root/include/python3.12/patchlevel.h",
                       "#define PY_MAJOR_VERSION 3\n#define PY_MINOR_VERSION 11\n")
            with self.assertRaisesRegex(ValueError, "PY_MINOR_VERSION 12"):
                build.plan(self.args)

    def test_cli_dry_run_emits_plan_without_running_or_creating_outputs(self):
        argv = []
        for key, value in vars(self.args).items():
            if key != "dry_run":
                argv += ["--" + key.replace("_", "-"), str(value)]
        argv.append("--dry-run")
        before = self.snapshot()
        output = io.StringIO()
        with patch("subprocess.run", side_effect=AssertionError("dry run invoked a tool")), contextlib.redirect_stdout(output):
            self.assertEqual(build.main(argv), 0)
        self.assertEqual(json.loads(output.getvalue())["platform"], "linux-x86_64")
        self.assertEqual(self.snapshot(), before)

    def test_invalid_platform_profile_or_tool_has_no_build_side_effects(self):
        with patch("subprocess.run", side_effect=AssertionError("invalid plan invoked a tool")):
            for attribute, invalid in (("platform", "windows-x86_64"), ("platform", "linux-aarch64"),
                                       ("config", "relwithdebinfo"), ("protoc", str(self.root / "missing-protoc"))):
                with self.subTest(attribute=attribute, value=invalid):
                    original = getattr(self.args, attribute)
                    setattr(self.args, attribute, invalid)
                    before = self.snapshot()
                    with self.assertRaises((ValueError, OSError)):
                        build.plan(self.args)
                    self.assertEqual(self.snapshot(), before)
                    setattr(self.args, attribute, original)
            for relative, content in (("sdks/usd_root/include/pxr/pxr.h", "#define PXR_VERSION 2505\n"),
                                      ("sdks/python_root/include/python3.12/patchlevel.h", "#define PY_MAJOR_VERSION 3\n#define PY_MINOR_VERSION 11\n")):
                path = self.root / relative
                original = path.read_bytes()
                path.write_text(content)
                before = self.snapshot()
                with self.assertRaises(ValueError):
                    build.plan(self.args)
                self.assertEqual(self.snapshot(), before)
                path.write_bytes(original)
            with not_executable(Path(self.args.grpc_cpp_plugin)), self.assertRaises(ValueError):
                build.plan(self.args)
        self.assertFalse(Path(self.args.build_dir).exists())
        self.assertFalse(Path(self.args.target_dir).exists())

    def owned_stage(self):
        stage = Path(self.args.target_dir) / "usd/omniverse/live"
        stage.mkdir(parents=True)
        (stage / "package-manifest.json").write_text(json.dumps({"schema_version": 1, "owner": "native-live-build"}))
        (stage / "retained-old-bundle").write_bytes(b"prior package bytes")
        return stage

    def test_configure_and_build_failures_do_not_replace_existing_stage(self):
        stage = self.owned_stage()
        before = self.snapshot(stage)
        for fail_at in (1, 2):
            calls = []
            def fail(command, **kwargs):
                calls.append(command)
                if len(calls) == fail_at:
                    raise subprocess.CalledProcessError(23, command)
            with self.subTest(fail_at=fail_at), patch("subprocess.run", side_effect=fail):
                plan = build.plan(self.args)
                with self.assertRaises(subprocess.CalledProcessError):
                    build.execute(plan)
                self.assertEqual(len(calls), fail_at)
                self.assertEqual(self.snapshot(stage), before)
                self.assertFalse(list(stage.parent.glob(".live-stage-*")))

    def test_foreign_cmake_cache_rejects_before_tool_invocation(self):
        plan = build.plan(self.args)
        cache = Path(self.args.build_dir)
        cache.mkdir()
        (cache / "CMakeCache.txt").write_text("foreign cache must remain")
        with patch("subprocess.run", side_effect=AssertionError("foreign cache invoked a tool")):
            with self.assertRaises(ValueError):
                build.execute(plan)
        self.assertEqual((cache / "CMakeCache.txt").read_text(), "foreign cache must remain")
        self.assertFalse(Path(plan["target"]).exists())

    def test_changed_tool_contents_require_fresh_cache_even_at_same_path(self):
        plan = build.plan(self.args)
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(23, "synthetic-configure")):
            with self.assertRaises(subprocess.CalledProcessError):
                build.execute(plan)
        Path(self.args.protoc).write_text("same path, changed generator contents")
        before = self.snapshot()
        with patch("subprocess.run", side_effect=AssertionError("changed support identity reused build cache")):
            with self.assertRaises(ValueError):
                build.execute(plan)
        self.assertEqual(self.snapshot(), before)

    def test_sdk_bytecode_preserves_build_cache_stamp(self):
        plan = build.plan(self.args)
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(23, "synthetic-configure")):
            with self.assertRaises(subprocess.CalledProcessError):
                build.execute(plan)
        stamp = Path(self.args.build_dir) / ".live-build-inputs.json"
        original_stamp = stamp.read_bytes()
        for sdk in (self.args.usd_root, self.args.python_root):
            for name in ("lib/__pycache__/module.cpython-312.pyc", "lib/module.pyo"):
                path = Path(sdk) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"generated SDK bytecode")
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(23, "synthetic-configure")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                build.execute(build.plan(self.args))
        self.assertEqual(run.call_count, 1)
        self.assertEqual(stamp.read_bytes(), original_stamp)
        header = Path(self.args.pybind11_root) / "include/pybind11/pybind11.h"
        header.write_bytes(header.read_bytes() + b" changed SDK input")
        with patch("subprocess.run", side_effect=AssertionError("changed SDK reused cache")):
            with self.assertRaisesRegex(ValueError, "Build cache inputs changed"):
                build.execute(build.plan(self.args))
        self.assertEqual(stamp.read_bytes(), original_stamp)

    def test_changed_python_interpreter_or_pybind_headers_cannot_reuse_build_cache(self):
        plan = build.plan(self.args)
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(23, "synthetic-configure")):
            with self.assertRaises(subprocess.CalledProcessError):
                build.execute(plan)
        stamp = Path(self.args.build_dir) / ".live-build-inputs.json"
        original_stamp = stamp.read_bytes()
        for path in (Path(self.args.python_root) / "bin/python3.12",
                     Path(self.args.pybind11_root) / "include/pybind11/pybind11.h"):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b" changed build dependency")
                before = self.snapshot()
                with patch("subprocess.run", side_effect=AssertionError("stale Python dependency reused cache")):
                    with self.assertRaises(ValueError):
                        build.execute(plan)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(stamp.read_bytes(), original_stamp)
                path.write_bytes(original)
                with patch("subprocess.run", side_effect=subprocess.CalledProcessError(23, "synthetic-configure")) as run:
                    with self.assertRaises(subprocess.CalledProcessError):
                        build.execute(build.plan(self.args))
                self.assertEqual(run.call_count, 1)

    def test_plan_preserves_selected_symlinks_and_retargeting_invalidates_cache(self):
        original_sdk, original_tool = Path(self.args.usd_root), Path(self.args.protoc)
        alternate_sdk = self.root / "alternate USD"
        shutil.copytree(original_sdk, alternate_sdk)
        (alternate_sdk / "lib/libusd_sdf.so").write_text("different SDK library contents")
        alternate_tool = self.write("host tools/alternate-protoc", "different generator contents")
        alternate_tool.chmod(0o755)
        selected_sdk, selected_tool = self.root / "selected USD", self.root / "selected protoc"
        selected_sdk.symlink_to(original_sdk, target_is_directory=True)
        selected_tool.symlink_to(original_tool)
        self.args.usd_root, self.args.protoc = str(selected_sdk), str(selected_tool)
        before = self.snapshot()
        with patch("subprocess.run", side_effect=AssertionError("planning invoked a tool")):
            plan = build.plan(self.args)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(plan["sdks"]["usd_root"], str(selected_sdk))
        self.assertEqual(plan["tools"]["protoc"], str(selected_tool))
        self.assertIn("-DLIVE_USD_ROOT=" + str(selected_sdk), plan["commands"][0])
        self.assertIn("-DLIVE_PROTOC=" + str(selected_tool), plan["commands"][0])
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(23, "synthetic-configure")):
            with self.assertRaises(subprocess.CalledProcessError):
                build.execute(plan)
        stamp = Path(self.args.build_dir) / ".live-build-inputs.json"
        original_stamp = stamp.read_bytes()
        recorded = json.loads(original_stamp)
        self.assertEqual(recorded["sdks"]["usd_root"], str(selected_sdk))
        self.assertEqual(recorded["tools"]["protoc"], str(selected_tool))
        self.assertEqual(recorded["support_identity"]["tools"]["protoc"]["sha256"],
                         hashlib.sha256(original_tool.read_bytes()).hexdigest())
        for selected, original, alternate in ((selected_sdk, original_sdk, alternate_sdk),
                                               (selected_tool, original_tool, alternate_tool)):
            with self.subTest(selection=selected.name):
                selected.unlink()
                selected.symlink_to(alternate, target_is_directory=alternate.is_dir())
                retargeted_plan = build.plan(self.args)
                self.assertEqual(retargeted_plan, plan)  # Selection stays fixed; contents changed.
                before = self.snapshot()
                with patch("subprocess.run", side_effect=AssertionError("retargeted support reused cache")):
                    with self.assertRaises(ValueError):
                        build.execute(retargeted_plan)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(stamp.read_bytes(), original_stamp)
                selected.unlink()
                selected.symlink_to(original, target_is_directory=original.is_dir())
                # Restoring the selected content permits the same cache to reach configuration.
                with patch("subprocess.run", side_effect=subprocess.CalledProcessError(23, "synthetic-configure")) as run:
                    with self.assertRaises(subprocess.CalledProcessError):
                        build.execute(build.plan(self.args))
                self.assertEqual(run.call_count, 1)

    def test_plan_rejects_unowned_destination_before_building(self):
        stage = Path(self.args.target_dir) / "usd/omniverse/live"
        stage.mkdir(parents=True)
        (stage / "package-manifest.json").write_text('{"schema_version": 1, "owner": "other-builder"}')
        before = self.snapshot(stage)
        with patch("subprocess.run", side_effect=AssertionError("unowned stage invoked a tool")):
            with self.assertRaises(ValueError):
                build.plan(self.args)
        self.assertEqual(self.snapshot(stage), before)
        self.assertFalse(Path(self.args.build_dir).exists())


if __name__ == "__main__":
    unittest.main()
