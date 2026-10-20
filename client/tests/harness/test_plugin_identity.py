# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import hashlib
import io
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.live import sanitizer
from tools.live import test as runner
from tools.live import plugin_identity
from tools.live.plugin_identity import inputs
from tools.live.wire_identity import USD_TREES, tree_hash
from tools.live.test_support import ProcessResult

SCRIPT = Path(__file__).resolve().parents[2] / "tools/live/plugin_identity.py"


class PluginIdentityTests(unittest.TestCase):
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
        package = self.root / "package"
        from tools.live import build, platforms
        host = platforms.host().name
        diagnostics_sample, participant_sample = build.samples(host)
        self.plugin = package / build.library(host)
        self.plugin.parent.mkdir(parents=True)
        self.plugin.write_bytes(b"synthetic shared plugin")
        self.plug_info = package / "resources/plugInfo.json"
        self.plug_info.parent.mkdir()
        real_root = Path(__file__).resolve().parents[2]
        self.plug_info.write_bytes(build.plug_info(real_root, host))
        self.dependencies = [self.root / name for name in ("libgrpc.a", "protoc", "grpc_cpp_plugin")]
        for path in self.dependencies:
            path.write_text("synthetic dependency " + path.name)
        for name in ("source/live/runtime/LiveMetadata.h", "source/live/scene/LiveUsd.cpp",
                     "source/live/scene/plugin/plugInfo.json.in", "tests/native/plugin/driver.cpp",
                     "tests/native/plugin/load_probe.cpp", "tests/reference_service/test_plugin.py",
                     "tests/reference_service/test_diagnostics_sample.py", "tests/native/diagnostics_log/driver.cpp",
                     "source/live/samples/diagnostics/main.cpp", "source/live/samples/diagnostics/DiagnosticLog.cpp",
                     "source/live/samples/diagnostics/DiagnosticLog.h",
                     "tools/live/test.py", "tools/live/reference.py", "tools/live/plugin_identity.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic source " + name)
        for name in inputs(self.root):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("synthetic source " + name)
        (self.root / "source/live/scene/plugin/plugInfo.json.in").write_bytes(self.plug_info.read_bytes())
        header_dir = package / "include/omni/usd/live"
        header_dir.mkdir(parents=True)
        for name in ("Live.h", "LiveUsd.h"):
            (header_dir / name).write_bytes((self.root / "include/omni/usd/live" / name).read_bytes())
        load_probe = package / "bin/load-probe"
        load_probe.parent.mkdir(exist_ok=True)
        load_probe.write_bytes(b"independent dynamic consumer")
        self.sample = package / diagnostics_sample
        self.sample.write_bytes(b"independent shared C ABI diagnostics consumer")
        participant = package / participant_sample
        participant.write_bytes(b"independent public USD participant")
        participant_tests = self.root / "live_participant_tests"
        participant_tests.write_bytes(b"standalone native log witnesses")
        formatter = self.root / "live_diagnostic_log_tests"
        formatter.write_bytes(b"standalone formatter checks")
        trees = []
        for role in ("flatbuffers_include", *USD_TREES):
            tree = self.root / "sdks" / role
            tree.mkdir(parents=True)
            (tree / "input").write_text("synthetic SDK " + role)
            trees.append({"role": role, "path": str(tree), "sha256": ""})
        flatbuffers = Path(trees[0]["path"]) / "flatbuffers"
        flatbuffers.mkdir()
        headers = {}
        for name in ("flatbuffers.h", "base.h", "stl_emulation.h"):
            path = flatbuffers / name
            path.write_text("synthetic pinned header " + name)
            headers[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance = {"flatbuffers": {"package": "synthetic", "version": "1", "headers": headers}}
        (self.root / "source/live/wire/schema").mkdir(parents=True, exist_ok=True)
        (self.root / "source/live/wire/schema/provenance.json").write_text(json.dumps(provenance))
        dependencies = [("plugin", self.plugin), ("plug_info", self.plug_info), ("load_probe", load_probe),
                        ("diagnostics_sample", self.sample), ("diagnostic_log_tests", formatter), ("participant_sample", participant), ("participant_tests", participant_tests)]
        dependencies += [("public_header", header_dir / name) for name in ("Live.h", "LiveUsd.h")]
        dependencies += [("native_dependency", path) for path in self.dependencies]
        for entry in trees:
            entry["sha256"] = tree_hash(Path(entry["path"]))
        self.manifest = {"schema_version": 1, "boundary": "plugin", "sanitizers": True,
            "compiler_version": "1", "artifact_sha256": hashlib.sha256(self.artifact.read_bytes()).hexdigest(),
            "flatbuffers": {"package": "synthetic", "version": "1"},
            "inputs": [{"path": name, "sha256": hashlib.sha256((self.root / name).read_bytes()).hexdigest()}
                       for name in inputs(self.root)],
            "dependencies": [{"role": role, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                             for role, path in dependencies], "dependency_trees": trees}
        self.manifest_path = Path(str(self.artifact) + ".manifest.json")
        self.manifest_path.write_text(json.dumps(self.manifest))
        module_patch = patch.object(plugin_identity, "__file__", str(self.root / "tools/live/plugin_identity.py"))
        module_patch.start()
        self.addCleanup(module_patch.stop)

    def test_full_source_and_dependency_identity(self):
        self.assertEqual(runner.native_identity(self.artifact, "plugin"), self.manifest)
        for name in inputs(self.root):
            with self.subTest(source=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                with self.assertRaisesRegex(ValueError, "stale source"):
                    runner.native_identity(self.artifact, "plugin")
                path.write_bytes(original)
        for path in (Path(entry["path"]) for entry in self.manifest["dependencies"]):
            with self.subTest(dependency=path):
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                with self.assertRaisesRegex(ValueError, "stale dependency"):
                    runner.native_identity(self.artifact, "plugin")
                path.write_bytes(original)
        cache = self.root / "tools/live/__pycache__/test.pyc"
        cache.parent.mkdir(); cache.write_bytes(b"ignored cache")
        self.assertEqual(runner.native_identity(self.artifact, "plugin"), self.manifest)
        (self.root / "source/live/scene/NewScene.h").write_text("new source")
        with self.assertRaisesRegex(ValueError, "missing required source"):
            runner.native_identity(self.artifact, "plugin")

    def test_diagnostics_roles_and_package_location_are_required(self):
        for role in ("diagnostics_sample", "diagnostic_log_tests", "participant_sample", "participant_tests"):
            entry = next(e for e in self.manifest["dependencies"] if e["role"] == role)
            for changed in ([e for e in self.manifest["dependencies"] if e["role"] != role],
                            self.manifest["dependencies"] + [entry]):
                with self.subTest(role=role, count=len(changed)):
                    self.manifest_path.write_text(json.dumps(dict(self.manifest, dependencies=changed)))
                    with self.assertRaisesRegex(ValueError, "package/dependency identity"):
                        runner.native_identity(self.artifact, "plugin")
        outside = self.root / "unrecorded-sample"
        outside.write_bytes(self.sample.read_bytes())
        changed = [dict(e, path=str(outside)) if e["role"] == "diagnostics_sample" else e
                   for e in self.manifest["dependencies"]]
        self.manifest_path.write_text(json.dumps(dict(self.manifest, dependencies=changed)))
        with self.assertRaisesRegex(ValueError, "sample outside recorded package"):
            runner.native_identity(self.artifact, "plugin")
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.assertEqual(runner.native_identity(self.artifact, "plugin"), self.manifest)
        for name in ("source/live/samples/diagnostics/main.cpp", "source/live/samples/diagnostics/DiagnosticLog.cpp",
                     "tests/native/diagnostics_log/driver.cpp"):
            self.assertIn(name, inputs(self.root))

    def test_sdk_roles_binary_boundary_and_metadata_checks(self):
        for entry in self.manifest["dependency_trees"]:
            path = Path(entry["path"]) / "input"
            original = path.read_bytes(); path.write_bytes(original + b" changed")
            with self.assertRaisesRegex(ValueError, "SDK differs"):
                runner.native_identity(self.artifact, "plugin")
            path.write_bytes(original)
            saved_tree = {p.relative_to(path.parent): p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
            shutil.rmtree(path.parent)
            with self.assertRaisesRegex(ValueError, "missing SDK tree"):
                runner.native_identity(self.artifact, "plugin")
            path.parent.mkdir()
            for relative, content in saved_tree.items():
                restored = path.parent / relative
                restored.parent.mkdir(parents=True, exist_ok=True)
                restored.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "unexpected native artifact"):
            runner.native_identity(self.artifact, "resolver_probe")
        for field, changed, phrase in (
                ("dependencies", self.manifest["dependencies"][1:], "package/dependency identity"),
                ("dependencies", self.manifest["dependencies"][:2], "package/dependency identity"),
                ("dependency_trees", self.manifest["dependency_trees"][:-1], "SDK identity")):
            self.manifest_path.write_text(json.dumps(dict(self.manifest, **{field: changed})))
            with self.assertRaisesRegex(ValueError, phrase):
                runner.native_identity(self.artifact, "plugin")
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.plug_info.write_text(json.dumps({"Plugins": [{"LibraryPath": "unrecorded.so"}]}))
        next(e for e in self.manifest["dependencies"] if e["role"] == "plug_info")["sha256"] = \
            hashlib.sha256(self.plug_info.read_bytes()).hexdigest()
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            runner.native_identity(self.artifact, "plugin")
        self.artifact.write_bytes(b"changed binary")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            runner.native_identity(self.artifact, "plugin")

    def test_runner_missing_capabilities_sanitizers_seed_and_replay(self):
        artifacts = self.root / "runs"
        executed = []

        def process(command, **options):
            executed.append((command, options["environment"].copy()))
            # The real lane writes completion evidence that the runner requires.
            (options["output"].parent / "plugin-reference-result.json").write_text(json.dumps(dict(
                schema_version=1, lane="plugin-reference", passed=True, discovered=1, tests_run=1, skipped=0,
                failures=0, errors=0, expected_failures=0, unexpected_successes=0,
                cases=[dict(id="synthetic.plugin-reference.fixture", outcome="passed", message="")])))
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

        status, missing = run(["--suite", "plugin-reference"])
        self.assertEqual(status, 2)
        self.assertIn("--plugin-driver", json.loads((missing / "manifest.json").read_text())["results"][0]["reason"])
        arguments = ["--suite", "plugin-reference", "--seed", "4287", "--plugin-driver", str(self.artifact)]
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
        self.assertEqual(replay["plugin_identity"], self.manifest)
        self.assertEqual(replay["sanitizer_runtime"], sanitizer.provenance())
        self.assertEqual(replay["plugin_driver"], str(self.artifact))
        self.assertEqual(executed[0][0], [sys.executable, str(self.root / "tools/live/reference.py"),
                         "--openssl", str(self.dependencies[1]), "--plugin-driver", str(self.artifact),
                         "--plugin-plug-info", str(self.plug_info), "--diagnostics-sample", str(self.sample)])
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
