# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The suite catalog agrees with the runner, CI and contracts, and native property and fuzz inputs are pinned."""
import ast
import json
from pathlib import Path
import re
import unittest

from tools.live import ci
from tools.live import test as runner

ROOT = Path(__file__).resolve().parents[2]
TAGS = {"pr", "sdk-free", "native", "reference", "campaign", "manual", "mandatory", "planned"}
PACKAGES = {"harness", "runtime", "wire", "scene", "resolver", "plugin", "python", "kit", "reference"}
# Entry points that apply LIVE_TEST_SCENARIOS and LIVE_TEST_SHARD: foundation.py and reference.py's discovered lanes.
CASE_SELECTING = {"foundation"} | runner.REFERENCE_LANES - {"kit-reference"}
# Angle-bracket include roots of native tests, and the pinned input that provides each (deps/README.md).
PINNED_INCLUDES = {"pxr": "USD", "grpcpp": "grpc", "openssl": "openssl", "omni": "repository", "SHA256.h": "repository"}
POSIX_HEADERS = {"dirent.h", "dlfcn.h", "execinfo.h", "fcntl.h", "pthread.h", "unistd.h", "sys", "mach-o"}
# The Windows SDK headers the Windows branches of the native tests use, as the POSIX branches use the system's.
WINDOWS_HEADERS = {"windows.h", "tlhelp32.h", "psapi.h", "io.h"}


class SuiteCatalogTests(unittest.TestCase):
    def test_catalog_names_every_suite_once(self):
        suites = runner.CATALOG["suites"]
        self.assertEqual(set(suites), set(runner.SUITES) | runner.PLANNED)
        self.assertFalse(set(runner.SUITES) & runner.PLANNED)
        self.assertTrue(runner.PLANNED)

    def test_entries_are_well_formed(self):
        contracts = {entry["id"] for entry in
                     json.loads((ROOT / "tests/contracts/catalog.json").read_text())["contracts"]}
        for name, entry in runner.CATALOG["suites"].items():
            with self.subTest(suite=name):
                self.assertIn(entry["package"], PACKAGES)
                self.assertTrue(entry["tags"])
                self.assertLessEqual(set(entry["tags"]), TAGS)
                self.assertEqual("planned" in entry["tags"], bool(entry.get("planned")))
                self.assertTrue(entry["fixtures"])
                for fixture in entry["fixtures"]:
                    self.assertTrue((ROOT / fixture).exists(), fixture)
                self.assertTrue(entry["contracts"])
                self.assertLessEqual(set(entry["contracts"]), contracts)
                self.assertEqual(set(entry["budget"]), {"artifact_bytes", "artifact_files"})
                self.assertTrue(all(type(value) is int and value > 0 for value in entry["budget"].values()))
                self.assertLessEqual(set(entry["redact"]), set(runner.CATALOG["redaction"]))
        for pattern in runner.CATALOG["redaction"].values():
            re.compile(pattern)


    def test_case_selection_is_declared_exactly_where_the_entry_point_applies_it(self):
        declared = {name for name, entry in runner.CATALOG["suites"].items() if entry.get("case_selection")}
        self.assertEqual(declared, CASE_SELECTING)

    def test_driver_modes_are_the_native_probe_modes(self):
        declared = {name: entry["driver_modes"] for name, entry in runner.CATALOG["suites"].items() if "driver_modes" in entry}
        tree = ast.parse((ROOT / "tests/reference_service/test_native.py").read_text())
        probe = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "PROBE_MODES" for target in node.targets))
        self.assertEqual(declared, {"native-reference": sorted(ast.literal_eval(probe))})
        driver = (ROOT / "tests/native/runtime/layer_actor_driver.cpp").read_text()
        for mode in declared["native-reference"]:
            self.assertIn(f'"{mode}"', driver)

    def test_reference_lanes_redact_fixture_credentials(self):
        for name in runner.REFERENCE_LANES | {"writer-fuzz"}:
            self.assertEqual(set(runner.CATALOG["suites"][name]["redact"]), {"bearer", "pem-private-key"}, name)


class NativePinTests(unittest.TestCase):
    """Native property and fuzz tests draw every input from a pinned source (deps/README.md#test-dependencies)."""

    def sources(self):
        return sorted(path for suffix in ("*.c", "*.cpp", "*.h") for path in (ROOT / "tests/native").rglob(suffix))

    def test_third_party_includes_come_from_pinned_inputs(self):
        readme = (ROOT / "deps/README.md").read_text()
        manifests = "".join(path.read_text() for path in sorted((ROOT / "deps").glob("*.packman.xml")))
        standard = set()
        unpinned = []
        for path in self.sources():
            for header in re.findall(r"^\s*#\s*include\s*<([^>]+)>", path.read_text(errors="replace"), re.M):
                root = header.split("/")[0]
                if root in PINNED_INCLUDES or root in POSIX_HEADERS or root in WINDOWS_HEADERS:
                    continue
                if "/" not in header and (re.fullmatch(r"[a-z_]+", header) or re.fullmatch(r"(assert|stddef|string|stdint|stdio|stdlib)\.h", header)):
                    standard.add(header)
                    continue
                unpinned.append((path.relative_to(ROOT).as_posix(), header))
        self.assertEqual(unpinned, [])
        self.assertIn("random", standard)
        for root, pin in PINNED_INCLUDES.items():
            if pin == "repository":
                continue
            self.assertTrue(f'<package name="{pin}"' in manifests or f"| {pin} " in readme, (root, pin))
        self.assertIn("## Test dependencies", readme)

    def test_seeded_generation_uses_only_standard_fixed_engine_output(self):
        # std distributions and std::shuffle draw in implementation-defined ways; the engines' output is fixed.
        found = [(path.relative_to(ROOT).as_posix(), match) for path in self.sources()
                 for match in re.findall(r"std::\w+_distribution|std::shuffle|std::random_shuffle|std::sample|std::random_device",
                                         path.read_text(errors="replace"))]
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
