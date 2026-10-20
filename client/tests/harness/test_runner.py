# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
MACOS_SUITES = {"foundation", "uri", "wire-envelope", "lock-noalloc"}


class RunnerTests(unittest.TestCase):
    def assert_capability_rejection(self, result, suite, missing):
        self.assertEqual(result["outcome"], "infrastructure_error")
        if sys.platform == "darwin" and suite not in MACOS_SUITES:
            self.assertIn(suite + " is unsupported on macOS", result["reason"])
        else:
            self.assertIn(missing, result["reason"])

    def test_reference_identity_and_execution_use_same_clean_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe = ("import json, os\n"
                     "print(json.dumps({key: os.environ.get(key) for key in "
                     "['PYTHONPATH', 'PYTHONHOME', 'PYTHONNOUSERSITE', 'SERVICE_SECRET']}))\n")
            if os.name == "posix":
                # An executable script that ignores the arguments an interpreter would get.
                interpreter = root / "identity-probe"
                interpreter.write_text("#!" + sys.executable + "\n" + probe)
                interpreter.chmod(0o700)
            else:
                # Windows runs a command script, not a shebang; it runs the probe and drops the arguments.
                (root / "identity-probe.py").write_text(probe)
                interpreter = root / "identity-probe.cmd"
                interpreter.write_text(f'@"{sys.executable}" "%~dp0identity-probe.py"\r\n')
            artifacts = root / "artifacts"
            environment = dict(os.environ, PYTHONPATH="ambient-shadow", SERVICE_SECRET="synthetic-secret")
            result = subprocess.run([sys.executable, str(ROOT / "tools/live/test.py"),
                                     "--suite", "reference-service", "--seed", "42",
                                     "--reference-python", str(interpreter), "--reference-openssl", str(interpreter),
                                     "--artifact-dir", str(artifacts)], env=environment,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            directory = next(artifacts.iterdir())
            manifest = json.loads((directory / "manifest.json").read_text())
            if sys.platform == "darwin":
                self.assert_capability_rejection(manifest["results"][0], "reference-service", "")
                self.assertIsNone(manifest["reference_identity"])
                self.assertFalse((directory / "0.log").exists())
                return
            # The shim exits without reporting readiness: a failed service start, never completion evidence.
            self.assertEqual((manifest["results"][0]["outcome"], manifest["results"][0]["reason"]),
                             ("service_start_failed", "reference service did not start: exited with 0 before reporting readiness"))
            observed = json.loads((directory / "0.log").read_text())
            self.assertEqual(manifest["reference_identity"], observed)
            self.assertEqual(observed, {"PYTHONPATH": str(ROOT), "PYTHONHOME": None,
                                        "PYTHONNOUSERSITE": "1", "SERVICE_SECRET": None})

    def test_native_reference_requires_explicit_driver(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, str(ROOT / "tools/live/test.py"),
                                     "--suite", "native-reference", "--seed", "42", "--artifact-dir", temporary],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            manifest = json.loads((next(Path(temporary).iterdir()) / "manifest.json").read_text())
            self.assertFalse(manifest["passed"])
            self.assert_capability_rejection(manifest["results"][0], "native-reference", "--unary-driver")

    def test_reference_service_requires_explicit_provisioned_tools(self):
        for options, missing in (([], "--reference-python"),
                                 (["--reference-python", sys.executable], "--reference-openssl")):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                result = subprocess.run([sys.executable, str(ROOT / "tools/live/test.py"),
                                         "--suite", "reference-service", "--seed", "42", *options,
                                         "--artifact-dir", temporary], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                manifest = json.loads((next(Path(temporary).iterdir()) / "manifest.json").read_text())
                self.assertFalse(manifest["passed"])
                self.assert_capability_rejection(manifest["results"][0], "reference-service", missing)

    def test_native_components_require_explicit_built_driver(self):
        for suite, option in (("lock-noalloc", "--lock-driver"), ("runtime-foundation", "--runtime-driver"),
                              ("sdf-scene", "--sdf-driver"),
                              ("wire-envelope", "--wire-driver"), ("plugin-reference", "--plugin-driver")):
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as temporary:
                result = subprocess.run([sys.executable, str(ROOT / "tools/live/test.py"),
                                         "--suite", suite, "--seed", "42", "--artifact-dir", temporary],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                directory = next(Path(temporary).iterdir())
                manifest = json.loads((directory / "manifest.json").read_text())
                self.assertFalse(manifest["passed"])
                self.assert_capability_rejection(manifest["results"][0], suite, option)

    def test_missing_product_suite_cannot_pass_and_replay_keeps_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Path(temporary)
            result = subprocess.run([sys.executable, str(ROOT / "tools/live/test.py"),
                                     "--suite", "transport", "--seed", "42", "--clients", "3",
                                     "--artifact-dir", str(artifacts)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            first = next(artifacts.iterdir())
            manifest = json.loads((first / "manifest.json").read_text())
            self.assertFalse(manifest["passed"])
            self.assertEqual(manifest["results"][0]["outcome"], "infrastructure_error")
            xml = ET.parse(first / "junit.xml").getroot()
            self.assertEqual((xml.tag, xml.attrib["errors"]), ("testsuites", "1"))
            self.assertEqual([suite.get("name") for suite in xml], ["transport"])
            replay = subprocess.run([sys.executable, str(ROOT / "tools/live/test.py"), "--replay",
                                     str(first / "replay.json"), "--artifact-dir", str(artifacts)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(replay.returncode, 2, replay.stdout + replay.stderr)
            second = next(path for path in artifacts.iterdir() if path != first)
            self.assertEqual(json.loads((first / "seeds.json").read_text()),
                             json.loads((second / "seeds.json").read_text()))

    def test_invalid_replay_or_selection_fails_before_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.json"
            path.write_text('{"schema_version": 9000}')
            for options in (("--replay", str(path)), ("--suite", "foundation,foundation"),
                            ("--seed", "-1"), ("--keep-runs", "1")):
                result = subprocess.run([sys.executable, str(ROOT / "tools/live/test.py"), *options,
                                         "--artifact-dir", temporary], capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 2)
            self.assertEqual(list(Path(temporary).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
