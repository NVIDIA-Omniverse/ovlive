# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python artifact replay must include wrappers, SDK and the shared runtime."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.live import python_identity as identity


class InterpreterProbeTests(unittest.TestCase):
    def test_isolated_probe_does_not_write_bytecode_even_into_a_cold_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'cold-cache'
            actual_run = subprocess.run

            def run_with_cold_cache(command, **kwargs):
                # Execute the real probe with an empty bytecode search location;
                # pre-existing system caches must not mask an attempted write.
                return actual_run([command[0], '-X', 'pycache_prefix=' + str(prefix), *command[1:]], **kwargs)

            with patch.object(identity.subprocess, 'run', side_effect=run_with_cold_cache):
                record = identity.interpreter_identity(Path(sys.executable))
            self.assertEqual(record['sha256'], identity.digest(sys.executable))
            self.assertIn(sys.version.split()[0], record['version_abi'][0])
            self.assertFalse(list(prefix.rglob('*.pyc')))


class PythonIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source/bindings/python/omni/usd/live"
        self.package = self.root / "build/package"
        self.artifact = self.package / "python/omni/usd/live/_live.cpython-312-x86_64-linux-gnu.so"
        self.pybind = self.root / "sdk/include"
        for name in ("cmake/LivePython.cmake", "tools/live/python_identity.py",
                     "source/bindings/python/omni/usd/live/Native.cpp",
                     "source/bindings/python/omni/usd/live/__init__.py", "sdk/include/pybind11.h"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        self.artifact.parent.mkdir(parents=True)
        self.artifact.write_bytes(b"synthetic extension, never executed")
        (self.artifact.parent / "__init__.py").write_bytes((self.source / "__init__.py").read_bytes())
        kit_files = ("config/extension.toml", "python/omni/kit/live_services/__init__.py",
                     "python/omni/kit/live_services/_bootstrap.py")
        for name in kit_files:
            source = self.root / "source/kit" / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("synthetic Kit " + name)
            destination = self.package / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        (self.package / "config/live-sdk.json").write_text(json.dumps(dict(schema_version=1, usd_profile="kit-25.11")))
        self.native = {"dependencies": [{"role": "plugin", "path": str(self.package / "lib/libomni_usd_live.so")}],
                       "marker": "independent native identity"}
        self.interpreter = {"path": "/explicit/test/interpreter", "marker": "independent interpreter identity"}
        self.record = {"schema_version": 1, "boundary": "python_adapter",
                       "artifact_sha256": identity.digest(self.artifact), "sources": identity.sources(self.root),
                       "modules": identity.modules(self.artifact.parent), "plugin_driver": "/test/driver",
                       "plugin_identity": dict(self.native), "pybind_include": str(self.pybind),
                       "pybind_sha256": identity.tree_hash(self.pybind), "interpreter": dict(self.interpreter)}
        self.record["kit_configuration"] = {name: identity.digest(self.package / name)
                                            for name in (*kit_files, "config/live-sdk.json")}
        self.save()
        for p in (patch.object(identity, "ROOT", self.root),
                  patch("tools.live.test.native_identity", side_effect=lambda *args: self.native),
                  patch.object(identity, "interpreter_identity", side_effect=lambda *args: self.interpreter)):
            p.start()
            self.addCleanup(p.stop)
        identity.validate(self.artifact)

    def save(self):
        Path(str(self.artifact) + ".manifest.json").write_text(json.dumps(self.record))

    def test_wrapper_changes_and_added_native_source_reject(self):
        wrapper = self.artifact.parent / "__init__.py"
        original = wrapper.read_bytes()
        wrapper.write_bytes(original + b"\nchanged")
        with self.assertRaisesRegex(ValueError, "wrappers"):
            identity.validate(self.artifact)
        wrapper.write_bytes(original)
        identity.validate(self.artifact)
        (self.source / "New.cpp").write_text("new implementation")
        with self.assertRaisesRegex(ValueError, "sources"):
            identity.validate(self.artifact)

    def test_binary_or_underlying_plugin_change_reject(self):
        self.native = {**self.native, "marker": "other runtime"}
        with self.assertRaisesRegex(ValueError, "plugin dependency"):
            identity.validate(self.artifact)
        self.native = self.record["plugin_identity"]
        identity.validate(self.artifact)
        self.artifact.write_bytes(b"different extension")
        with self.assertRaisesRegex(ValueError, "artifact/sources"):
            identity.validate(self.artifact)

    def test_sdk_and_interpreter_changes_reject(self):
        header = self.pybind / "pybind11.h"
        original = header.read_bytes()
        header.write_bytes(b"changed ABI headers")
        with self.assertRaisesRegex(ValueError, "pybind SDK"):
            identity.validate(self.artifact)
        header.write_bytes(original)
        identity.validate(self.artifact)
        self.interpreter = {**self.interpreter, "marker": "different ABI"}
        with self.assertRaisesRegex(ValueError, "interpreter"):
            identity.validate(self.artifact)

    def test_kit_configuration_profile_and_module_routing_are_bound(self):
        profile = self.package / "config/live-sdk.json"
        original = profile.read_bytes()
        profile.write_text(json.dumps(dict(schema_version=1, usd_profile="stock-25.11")))
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            identity.validate(self.artifact)
        profile.write_text(json.dumps(dict(schema_version=1, usd_profile="unknown")))
        with self.assertRaisesRegex(ValueError, "SDK profile"):
            identity.validate(self.artifact)
        profile.write_bytes(original)
        (self.package / "config/extension.toml").write_text("wrong Python module")
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            identity.validate(self.artifact)

    def test_rehashed_redirected_plugin_is_not_canonical_package(self):
        self.native = {"dependencies": [{"role": "plugin", "path": "/other/package/lib/libomni_usd_live.so"}]}
        self.record["plugin_identity"] = self.native
        self.save()
        with self.assertRaisesRegex(ValueError, "canonical plugin package"):
            identity.validate(self.artifact)

    def test_unhashed_bytecode_cannot_override_recorded_wrapper_source(self):
        cache = self.artifact.parent / "__pycache__/__init__.cpython-312.pyc"
        cache.parent.mkdir()
        cache.write_bytes(b"unchecked compiled wrapper")
        with self.assertRaisesRegex(ValueError, "bytecode"):
            identity.validate(self.artifact)
        cache.unlink()
        identity.validate(self.artifact)


class PythonRunnerCapabilityTests(unittest.TestCase):
    def test_missing_extension_is_recorded_as_infrastructure_failure(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(root / "tools/live/test.py"),
                "--suite", "python-reference", "--seed", "42", "--artifact-dir", directory],
                cwd=root, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            runs = list(Path(directory).glob("run-*"))
            self.assertEqual(len(runs), 1)
            manifest = json.loads((runs[0] / "manifest.json").read_text())
            self.assertFalse(manifest["passed"])
            self.assertEqual(manifest["results"][0]["outcome"], "infrastructure_error")
            reason = manifest["results"][0]["reason"]
            self.assertIn("python-reference is unsupported on macOS" if sys.platform == "darwin"
                          else "--python-extension", reason)
            if sys.platform == "darwin":
                self.assertIsNone(manifest["python_binding_identity"])
                self.assertFalse((runs[0] / "0.log").exists())
            self.assertTrue((runs[0] / "seeds.json").is_file())
