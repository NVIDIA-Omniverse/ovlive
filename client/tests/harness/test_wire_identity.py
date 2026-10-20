# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.live.test import native_identity
from tools.live import sanitizer
from tools.live import test as runner
from tools.live.test_support import ProcessResult
from tools.live.wire_identity import inputs, tree_hash

SCRIPT = Path(__file__).resolve().parents[2] / "tools/live/wire_identity.py"


class WireIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        support = self.root / "tests/native/support/test_tls.h"  # Every native driver records the shared test support.
        support.parent.mkdir(parents=True)
        support.write_text("synthetic test support")
        self.sdk = self.root / "sdk"
        self.include = self.sdk / "include"
        headers = {}
        for name in ("flatbuffers.h", "base.h", "stl_emulation.h"):
            path = self.include / "flatbuffers" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic SDK " + name)
            headers[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ("source/live/wire/OmpfEnvelope.cpp", "source/live/wire/OmpfEnvelope.h",
                     "source/live/wire/schema/provenance.json", "source/live/wire/schema/schema.fbs",
                     "tests/native/wire/envelope.cpp"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic source " + name)
        for name in inputs(self.root):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("synthetic source " + name)
        (self.root / "source/live/wire/schema/provenance.json").write_text(json.dumps({
            "flatbuffers": {"package": "flatbuffers", "version": "synthetic-pin", "headers": headers}}))
        self.artifact = self.root / "wire-driver"
        self.artifact.write_bytes(b"synthetic native artifact")
        result = subprocess.run([sys.executable, str(SCRIPT), "--root", str(self.root),
                                 "--artifact", str(self.artifact), "--flatbuffers", str(self.sdk),
                                 "--compiler-id", "Synthetic", "--compiler-version", "1",
                                 "--configuration", "Debug", "--sanitizers", "1"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.manifest_path = Path(str(self.artifact) + ".manifest.json")
        self.manifest = json.loads(self.manifest_path.read_text())
        root_patch = patch("tools.live.test.ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_manifest_generation_and_source_closure(self):
        self.assertEqual(native_identity(self.artifact, "wire_envelope"), self.manifest)
        self.assertTrue(self.manifest["sanitizers"])
        self.assertEqual(self.manifest["sanitizer"], "address")
        self.assertEqual(self.manifest["compiler_id"], "Synthetic")
        self.assertTrue({"source/live/wire/schema/schema.fbs", "source/live/wire/OmpfEnvelope.cpp",
                         "source/live/support/SHA256.h", "source/live/support/SHA256.cpp",
                         "tools/live/test.py", "tools/live/process_guard.py",
                         "tools/live/sanitizer.py"}.issubset(inputs(self.root)))
        for name in ("source/live/wire/OmpfEnvelope.cpp", "source/live/wire/schema/schema.fbs",
                     "source/live/support/SHA256.h", "tools/live/test.py", "tools/live/sdk_tree_identity.py"):
            with self.subTest(source=name):
                source = self.root / name
                original = source.read_bytes()
                source.write_bytes(original + b" changed")
                with self.assertRaisesRegex(ValueError, "stale source"):
                    native_identity(self.artifact, "wire_envelope")
                source.write_bytes(original)
        (self.root / "source/live/wire/new.h").write_text("new input")
        with self.assertRaisesRegex(ValueError, "missing required source"):
            native_identity(self.artifact, "wire_envelope")

    def test_sdk_changes_removals_and_pin_are_checked(self):
        for name in ("flatbuffers.h", "base.h", "stl_emulation.h"):
            with self.subTest(header=name):
                path = self.include / "flatbuffers" / name
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                with self.assertRaisesRegex(ValueError, "wire SDK differs"):
                    native_identity(self.artifact, "wire_envelope")
                # Rehashing a changed SDK cannot bypass its provenance pin.
                self.manifest["dependency_trees"][0]["sha256"] = tree_hash(self.include)
                self.manifest_path.write_text(json.dumps(self.manifest))
                with self.assertRaisesRegex(ValueError, "provenance pin"):
                    native_identity(self.artifact, "wire_envelope")
                path.write_bytes(original)
                self.manifest["dependency_trees"][0]["sha256"] = tree_hash(self.include)
                self.manifest_path.write_text(json.dumps(self.manifest))
        extra = self.include / "additional.h"
        extra.write_text("unexpected new SDK input")
        with self.assertRaisesRegex(ValueError, "wire SDK differs"):
            native_identity(self.artifact, "wire_envelope")
        extra.unlink()
        (self.include / "flatbuffers/base.h").unlink()
        with self.assertRaisesRegex(ValueError, "wire SDK differs"):
            native_identity(self.artifact, "wire_envelope")

    def test_binary_boundary_and_missing_sdk_rejected(self):
        with self.assertRaisesRegex(ValueError, "unexpected native artifact"):
            native_identity(self.artifact, "sdf_scene")
        self.manifest["dependency_trees"] = []
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "missing FlatBuffers"):
            native_identity(self.artifact, "wire_envelope")
        self.artifact.write_bytes(b"replaced native artifact")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            native_identity(self.artifact, "wire_envelope")

    def test_runner_passes_seed_sanitizers_and_enforces_replay_identity(self):
        artifacts = self.root / "runs"
        executed = []

        def process(command, **options):
            executed.append((command, options["environment"].copy()))
            options["output"].write_text("master_seed=12345 scenario=ompf-envelope\nlive-cases passed=6\n")
            return ProcessResult("passed", 0, 1, 0, False, True)

        def run(arguments):
            with patch.object(sys, "argv", ["test.py", *arguments, "--artifact-dir", str(artifacts)]), \
                    patch.object(runner, "fingerprint", return_value="synthetic-source-fingerprint"), \
                    patch.object(runner, "identity", return_value="synthetic-toolchain"), \
                    patch.object(runner, "run_process", side_effect=process), contextlib.redirect_stdout(io.StringIO()):
                return runner.main()

        self.assertEqual(run(["--suite", "wire-envelope", "--seed", "12345", "--wire-driver", str(self.artifact)]), 0)
        first = next(artifacts.iterdir())
        replay = json.loads((first / "replay.json").read_text())
        self.assertEqual(replay["wire_driver"], str(self.artifact))
        self.assertEqual(replay["wire_identity"], self.manifest)
        self.assertEqual(executed[0][0], [str(self.artifact), "12345"])
        self.assertEqual(executed[0][1]["LIVE_TEST_SEED"], "12345")
        self.assertEqual(executed[0][1]["ASAN_OPTIONS"], sanitizer.runtime_options()["ASAN_OPTIONS"])
        self.assertEqual(executed[0][1]["LSAN_OPTIONS"], sanitizer.runtime_options()["LSAN_OPTIONS"])
        self.assertEqual(executed[0][1]["UBSAN_OPTIONS"], "halt_on_error=1:print_stacktrace=1")
        self.assertEqual(run(["--replay", str(first / "replay.json")]), 0)
        self.assertEqual(executed[1][0], executed[0][0])
        self.manifest["compiler_version"] = "changed-build"
        self.manifest_path.write_text(json.dumps(self.manifest))
        previous = set(artifacts.iterdir())
        self.assertEqual(run(["--replay", str(first / "replay.json")]), 2)
        self.assertEqual(len(executed), 2)
        failed = next(iter(set(artifacts.iterdir()) - previous))
        result = json.loads((failed / "manifest.json").read_text())
        self.assertFalse(result["passed"])
        self.assertEqual(result["results"][0]["outcome"], "infrastructure_error")
        self.assertIn("wire artifact differs from replay", result["results"][0]["reason"])


    def test_runner_requires_the_driver_case_count(self):
        artifacts = self.root / "case-count-runs"
        for log in ('master_seed=7 scenario=ompf-envelope\n', 'live-cases passed=5\n', 'live-cases passed=6\nlate output\n'):
            with self.subTest(log=log):
                def process(command, **options):
                    options["output"].write_text(log)
                    return ProcessResult("passed", 0, 1, 0, False, True)
                before = set(artifacts.iterdir()) if artifacts.exists() else set()
                with patch.object(sys, "argv", ["test.py", "--suite", "wire-envelope", "--seed", "7",
                                                "--wire-driver", str(self.artifact), "--artifact-dir", str(artifacts)]), \
                        patch.object(runner, "fingerprint", return_value="synthetic-source"), \
                        patch.object(runner, "identity", return_value="synthetic-toolchain"), \
                        patch.object(runner, "run_process", side_effect=process), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(), 2)
                run = next(iter(set(artifacts.iterdir()) - before))
                result = json.loads((run / "manifest.json").read_text())["results"][0]
                self.assertEqual(result["outcome"], "infrastructure_error")
                self.assertIn("live-cases passed=6", result["reason"])


if __name__ == "__main__":
    unittest.main()
