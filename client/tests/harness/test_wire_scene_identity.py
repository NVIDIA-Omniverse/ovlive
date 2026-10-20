# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from tools.live import sanitizer
from tools.live import test as runner
from tools.live.test import native_identity
from tools.live.test_support import ProcessResult
from tools.live.wire_identity import inputs, tree_hash, USD_TREES
from tests.harness import test_wire_identity as wire_tests


class WireSceneIdentityTests(unittest.TestCase):
    def setUp(self):
        wire_tests.WireIdentityTests.setUp(self)
        self.sdks = {name: self.root / name for name in ("usd", "python")}
        for role in USD_TREES:
            sdk, directory = role.split("_")
            tree = self.sdks[sdk] / directory
            tree.mkdir(parents=True, exist_ok=True)
            (tree / "input").write_text("synthetic " + role)
        for name in ("source/live/scene/WireScene.cpp", "source/live/scene/WireScene.h",
                     "source/live/scene/SdfSceneData.cpp", "source/live/support/Spec.cpp",
                     "source/live/support/Spec.h", "tests/native/scene/wire_scene.cpp"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic scene source " + name)
        for name in inputs(self.root, "wire_scene"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("synthetic " + name)
        result = subprocess.run([
            sys.executable, str(wire_tests.SCRIPT), "--root", str(self.root), "--artifact", str(self.artifact),
            "--boundary", "wire_scene", "--flatbuffers", str(self.sdk),
            *[argument for name, path in self.sdks.items() for argument in ("--" + name, str(path))],
            "--compiler-id", "Synthetic", "--compiler-version", "1", "--configuration", "Debug",
            "--sanitizers", "1"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.manifest = json.loads(self.manifest_path.read_text())

    def test_source_and_build_closure(self):
        self.assertEqual(native_identity(self.artifact, "wire_scene"), self.manifest)
        for name in ("CMakeLists.txt", "cmake/LiveWire.cmake", "cmake/LiveSceneCore.cmake",
                     "cmake/LiveWireScene.cmake", "cmake/LiveUsdWire.cmake",
                     "source/live/scene/WireScene.cpp", "source/live/scene/WireScene.h",
                     "source/live/scene/SdfSceneData.cpp", "source/live/support/Spec.cpp",
                     "source/live/support/Spec.h", "source/live/support/SHA256.cpp",
                     "tests/native/scene/wire_scene.cpp", "tools/live/test.py", "tools/live/wire_identity.py",
                     "source/live/wire/OmpfEnvelope.cpp",
                     "source/live/wire/schema/schema.fbs", "tests/native/wire/envelope.cpp"):
            with self.subTest(input=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                with self.assertRaisesRegex(ValueError, "stale source"):
                    native_identity(self.artifact, "wire_scene")
                path.write_bytes(original)
        (self.root / "source/live/scene/new_graph.h").write_text("new graph source")
        with self.assertRaisesRegex(ValueError, "missing required source"):
            native_identity(self.artifact, "wire_scene")

    def test_every_sdk_tree_is_checked(self):
        self.assertEqual({entry["role"] for entry in self.manifest["dependency_trees"]},
                         {"flatbuffers_include", *USD_TREES})
        for entry in self.manifest["dependency_trees"]:
            with self.subTest(role=entry["role"]):
                path = next(p for p in Path(entry["path"]).rglob("*") if p.is_file())
                original = path.read_bytes()
                path.write_bytes(original + b" changed SDK")
                with self.assertRaisesRegex(ValueError, "SDK differs"):
                    native_identity(self.artifact, "wire_scene")
                path.unlink()
                with self.assertRaisesRegex(ValueError, "SDK differs"):
                    native_identity(self.artifact, "wire_scene")
                path.write_bytes(original)

    def test_sdk_roles_boundary_and_binary_cannot_be_omitted(self):
        with self.assertRaisesRegex(ValueError, "unexpected native artifact"):
            native_identity(self.artifact, "usd_values")
        original = self.manifest_path.read_text()
        self.manifest["dependency_trees"][-1] = self.manifest["dependency_trees"][0]
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "missing SDK identity"):
            native_identity(self.artifact, "wire_scene")
        self.manifest_path.write_text(original)
        self.artifact.write_bytes(b"replaced binary")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            native_identity(self.artifact, "wire_scene")

    def test_rehashed_flatbuffers_still_requires_provenance_pin(self):
        header = self.include / "flatbuffers/base.h"
        header.write_text("different FlatBuffers build")
        entry = next(e for e in self.manifest["dependency_trees"] if e["role"] == "flatbuffers_include")
        entry["sha256"] = tree_hash(self.include)
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "provenance pin"):
            native_identity(self.artifact, "wire_scene")

    def test_runner_missing_capability_seed_sanitizers_and_replay(self):
        artifacts = self.root / "runs"
        executed = []

        def process(command, **options):
            executed.append((command, options["environment"].copy()))
            return ProcessResult("passed", 0, 1, 0, False, True)

        def run(arguments):
            before = set(artifacts.iterdir()) if artifacts.exists() else set()
            # Synthetic workloads exercise an admitted Linux/Windows dispatch boundary.
            with patch.object(runner, "suite_platform_error", return_value=None), \
                    patch.object(sys, "argv", ["test.py", *arguments, "--artifact-dir", str(artifacts)]), \
                    patch.object(runner, "fingerprint", return_value="source-fingerprint"), \
                    patch.object(runner, "identity", return_value="toolchain"), \
                    patch.object(runner, "run_process", side_effect=process), contextlib.redirect_stdout(io.StringIO()):
                status = runner.main()
            return status, next(iter(set(artifacts.iterdir()) - before))

        status, missing = run(["--suite", "wire-scene"])
        self.assertEqual(status, 2)
        self.assertEqual(executed, [])
        failure = json.loads((missing / "manifest.json").read_text())["results"][0]
        self.assertEqual(failure["outcome"], "infrastructure_error")
        self.assertIn("--wire-scene-driver", failure["reason"])
        original_manifest = self.manifest_path.read_text()
        wrong_boundary = dict(self.manifest, boundary="usd_values")
        self.manifest_path.write_text(json.dumps(wrong_boundary))
        status, rejected = run(["--suite", "wire-scene", "--wire-scene-driver", str(self.artifact)])
        self.assertEqual(status, 2)
        self.assertEqual(executed, [])
        failure = json.loads((rejected / "manifest.json").read_text())["results"][0]
        self.assertIn("unexpected native artifact", failure["reason"])
        self.manifest_path.write_text(original_manifest)
        source = self.root / "source/live/scene/WireScene.cpp"
        original_source = source.read_text()
        source.write_text(original_source + " stale")
        status, rejected = run(["--suite", "wire-scene", "--wire-scene-driver", str(self.artifact)])
        self.assertEqual(status, 2)
        self.assertEqual(executed, [])
        failure = json.loads((rejected / "manifest.json").read_text())["results"][0]
        self.assertIn("stale source", failure["reason"])
        source.write_text(original_source)
        status, first = run(["--suite", "wire-scene", "--seed", "8765", "--wire-scene-driver", str(self.artifact)])
        self.assertEqual(status, 0)
        replay = json.loads((first / "replay.json").read_text())
        self.assertEqual(replay["wire_scene_identity"], self.manifest)
        self.assertEqual(replay["wire_scene_driver"], str(self.artifact))
        self.assertEqual(executed[0][0], [str(self.artifact), "8765"])
        self.assertEqual(executed[0][1]["LIVE_TEST_SEED"], "8765")
        self.assertEqual(executed[0][1]["ASAN_OPTIONS"], sanitizer.runtime_options()["ASAN_OPTIONS"])
        self.assertEqual(executed[0][1]["LSAN_OPTIONS"], sanitizer.runtime_options()["LSAN_OPTIONS"])
        self.assertEqual(executed[0][1]["UBSAN_OPTIONS"], "halt_on_error=1:print_stacktrace=1")
        self.assertEqual(run(["--replay", str(first / "replay.json")])[0], 0)
        self.assertEqual(executed[0][0], executed[1][0])
        for key in ("LIVE_TEST_SEED", "ASAN_OPTIONS", "LSAN_OPTIONS", "UBSAN_OPTIONS"):
            self.assertEqual(executed[0][1][key], executed[1][1][key])
        self.manifest["compiler_version"] = "different build"
        self.manifest_path.write_text(json.dumps(self.manifest))
        status, failed = run(["--replay", str(first / "replay.json")])
        self.assertEqual(status, 2)
        self.assertEqual(len(executed), 2)
        failure = json.loads((failed / "manifest.json").read_text())["results"][0]
        self.assertEqual(failure["outcome"], "infrastructure_error")
        self.assertIn("artifact differs from replay", failure["reason"])


if __name__ == "__main__":
    unittest.main()
