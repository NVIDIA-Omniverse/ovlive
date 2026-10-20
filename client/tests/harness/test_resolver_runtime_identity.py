# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.live import sanitizer
from tools.live import test as runner
from tools.live.resolver_runtime_identity import inputs, SDK_TREES
from tools.live.test_support import ProcessResult

SCRIPT = Path(__file__).resolve().parents[2] / "tools/live/resolver_runtime_identity.py"


class ResolverRuntimeIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        support = self.root / "tests/native/support/test_tls.h"  # Every native driver records the shared test support.
        support.parent.mkdir(parents=True)
        support.write_text("synthetic test support")
        root_patch = patch.object(runner, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.artifact = self.root / "driver"
        self.artifact.write_bytes(b"synthetic driver; never executed")
        self.plugin = self.root / "plugin.so"
        self.plugin.write_bytes(b"synthetic production component")
        self.plug_info = self.root / "plugInfo.json"
        self.plug_info.write_text(json.dumps({"Plugins": [{"LibraryPath": str(self.plugin)}]}))
        self.dependencies = [self.root / name for name in ("libgrpc.a", "protoc", "grpc_cpp_plugin")]
        for path in self.dependencies:
            path.write_text("synthetic dependency " + path.name)
        for name in ("source/live/runtime/LiveMetadata.h", "source/live/runtime/LiveFinite.cpp",
                     "source/live/runtime/Live.exports", "source/live/resolver/LiveResolver.cpp",
                     "source/live/resolver/plugInfo.json.in", "tests/native/resolver_runtime/driver.cpp",
                     "source/live/protos/nvidia/omniverse/live/v1alpha/live.proto",
                     "tests/reference_service/test_resolver.py", "tests/reference_service/service.py",
                     "tools/live/test.py", "tools/live/reference.py", "tools/live/resolver_runtime_identity.py",
                     "tools/live/resolver_identity.py", "tests/harness/test_resolver_runtime_identity.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic source " + name)
        for name in inputs(self.root):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("synthetic source " + name)
        self.sdks = {name: self.root / name for name in ("usd", "python")}
        for role in SDK_TREES:
            sdk, directory = role.split("_")
            tree = self.sdks[sdk] / directory
            tree.mkdir(parents=True, exist_ok=True)
            (tree / "input").write_text("synthetic SDK " + role)
        result = subprocess.run([
            sys.executable, str(SCRIPT), "--root", str(self.root), "--artifact", str(self.artifact),
            "--plugin", str(self.plugin), "--plug-info", str(self.plug_info),
            *[item for path in self.dependencies for item in ("--dependency", str(path))],
            *[item for name, path in self.sdks.items() for item in ("--" + name, str(path))],
            "--compiler-id", "Synthetic", "--compiler-version", "1", "--configuration", "Debug",
            "--sanitizers", "1"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.manifest_path = Path(str(self.artifact) + ".manifest.json")
        self.manifest = json.loads(self.manifest_path.read_text())

    def test_full_source_and_dependency_identity(self):
        self.assertEqual(runner.native_identity(self.artifact, "resolver_runtime"), self.manifest)
        for name in inputs(self.root):
            with self.subTest(source=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                with self.assertRaisesRegex(ValueError, "stale source"):
                    runner.native_identity(self.artifact, "resolver_runtime")
                path.write_bytes(original)
        for path in (self.plugin, self.plug_info, *self.dependencies):
            with self.subTest(dependency=path):
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                with self.assertRaisesRegex(ValueError, "stale dependency"):
                    runner.native_identity(self.artifact, "resolver_runtime")
                path.write_bytes(original)
        cache = self.root / "tools/live/__pycache__/test.pyc"
        cache.parent.mkdir(); cache.write_bytes(b"ignored cache")
        self.assertEqual(runner.native_identity(self.artifact, "resolver_runtime"), self.manifest)
        (self.root / "source/live/resolver/NewResolver.h").write_text("new source")
        with self.assertRaisesRegex(ValueError, "missing required source"):
            runner.native_identity(self.artifact, "resolver_runtime")

    def test_sdk_roles_binary_boundary_and_metadata_checks(self):
        for entry in self.manifest["dependency_trees"]:
            path = Path(entry["path"]) / "input"
            original = path.read_bytes(); path.write_bytes(original + b" changed")
            with self.assertRaisesRegex(ValueError, "SDK differs"):
                runner.native_identity(self.artifact, "resolver_runtime")
            path.unlink(); path.parent.rmdir()
            with self.assertRaisesRegex(ValueError, "SDK directory is missing"):
                runner.native_identity(self.artifact, "resolver_runtime")
            path.parent.mkdir(); path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "unexpected native artifact"):
            runner.native_identity(self.artifact, "resolver_probe")
        for field, changed, phrase in (
                ("dependencies", self.manifest["dependencies"][1:], "plugin/metadata identity"),
                ("dependencies", self.manifest["dependencies"][:2], "native dependency identity"),
                ("dependency_trees", self.manifest["dependency_trees"][:-1], "SDK identity")):
            self.manifest_path.write_text(json.dumps(dict(self.manifest, **{field: changed})))
            with self.assertRaisesRegex(ValueError, phrase):
                runner.native_identity(self.artifact, "resolver_runtime")
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.plug_info.write_text(json.dumps({"Plugins": [{"LibraryPath": "unrecorded.so"}]}))
        next(e for e in self.manifest["dependencies"] if e["role"] == "plug_info")["sha256"] = \
            hashlib.sha256(self.plug_info.read_bytes()).hexdigest()
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "unrecorded binary"):
            runner.native_identity(self.artifact, "resolver_runtime")
        self.artifact.write_bytes(b"changed binary")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            runner.native_identity(self.artifact, "resolver_runtime")

    def test_runner_missing_capabilities_sanitizers_seed_and_replay(self):
        artifacts = self.root / "runs"
        executed = []

        def process(command, **options):
            executed.append((command, options["environment"].copy()))
            # The real lane writes completion evidence that the runner requires.
            (options["output"].parent / "resolver-reference-result.json").write_text(json.dumps(dict(
                schema_version=1, lane="resolver-reference", passed=True, discovered=1, tests_run=1, skipped=0,
                failures=0, errors=0, expected_failures=0, unexpected_successes=0,
                cases=[dict(id="synthetic.resolver-reference.fixture", outcome="passed", message="")])))
            return ProcessResult("passed", 0, 1, 0, False, True)

        def run(arguments, tools=True):
            before = set(artifacts.iterdir()) if artifacts.exists() else set()
            if tools:
                arguments = [*arguments, "--reference-python", sys.executable, "--reference-openssl", str(self.dependencies[1])]
            # Synthetic workloads exercise an admitted Linux/Windows dispatch boundary.
            with patch.object(runner, "suite_platform_error", return_value=None), \
                    patch.object(sys, "argv", ["test.py", *arguments, "--artifact-dir", str(artifacts)]), \
                    patch.object(runner, "fingerprint", return_value="source-fingerprint"), \
                    patch.object(runner, "identity", return_value="toolchain"), \
                    patch.object(runner, "run_process", side_effect=process), \
                    patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout='{"reference":"synthetic"}')), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = runner.main()
            directory = next(iter(set(artifacts.iterdir()) - before))
            return status, directory

        status, missing = run(["--suite", "resolver-reference"])
        self.assertEqual(status, 2)
        self.assertIn("--resolver-runtime-driver", json.loads((missing / "manifest.json").read_text())["results"][0]["reason"])
        arguments = ["--suite", "resolver-reference", "--seed", "4287", "--resolver-runtime-driver", str(self.artifact)]
        status, missing = run(arguments, tools=False)
        self.assertEqual(status, 2)
        self.assertIn("--reference-python", json.loads((missing / "manifest.json").read_text())["results"][0]["reason"])
        self.assertEqual(executed, [])
        self.manifest_path.write_text(json.dumps(dict(self.manifest, boundary="resolver_probe")))
        status, rejected = run(arguments)
        self.assertEqual(status, 2)
        self.assertIn("unexpected native artifact", json.loads((rejected / "manifest.json").read_text())["results"][0]["reason"])
        self.manifest_path.write_text(json.dumps(self.manifest))
        source = self.root / "source/live/runtime/LiveMetadata.h"
        original = source.read_bytes(); source.write_bytes(original + b" changed")
        status, rejected = run(arguments)
        self.assertEqual(status, 2)
        self.assertIn("stale source", json.loads((rejected / "manifest.json").read_text())["results"][0]["reason"])
        source.write_bytes(original)
        self.assertEqual(executed, [])
        status, first = run(arguments)
        self.assertEqual(status, 0)
        replay = json.loads((first / "replay.json").read_text())
        self.assertEqual(replay["resolver_runtime_identity"], self.manifest)
        self.assertEqual(replay["resolver_runtime_driver"], str(self.artifact))
        self.assertEqual(executed[0][0], [sys.executable, str(self.root / "tools/live/reference.py"),
                         "--openssl", str(self.dependencies[1]), "--resolver-runtime-driver", str(self.artifact),
                         "--resolver-runtime-plugin", str(self.plugin), "--resolver-runtime-plug-info", str(self.plug_info)])
        self.assertEqual(executed[0][1]["LIVE_TEST_SEED"], "4287")
        self.assertEqual(executed[0][1]["ASAN_OPTIONS"], sanitizer.runtime_options()["ASAN_OPTIONS"])
        self.assertEqual(executed[0][1]["LSAN_OPTIONS"], sanitizer.runtime_options()["LSAN_OPTIONS"])
        self.assertEqual(executed[0][1]["UBSAN_OPTIONS"], "halt_on_error=1:print_stacktrace=1")
        for key in ("PXR_PLUGINPATH_NAME", "PXR_AR_DEFAULT_SEARCH_PATH", "AR_DEFAULT_RESOLVER"):
            self.assertEqual(executed[0][1][key], "")
        status, _ = run(["--replay", str(first / "replay.json")])
        self.assertEqual(status, 0)
        self.assertEqual(executed[0][0], executed[1][0])
        for key in ("LIVE_TEST_SEED", "ASAN_OPTIONS", "LSAN_OPTIONS", "UBSAN_OPTIONS"):
            self.assertEqual(executed[0][1][key], executed[1][1][key])
        self.manifest["compiler_version"] = "different build"
        self.manifest_path.write_text(json.dumps(self.manifest))
        status, failed = run(["--replay", str(first / "replay.json")])
        self.assertEqual(status, 2)
        self.assertEqual(len(executed), 2)
        self.assertIn("artifact differs from replay", json.loads((failed / "manifest.json").read_text())["results"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
