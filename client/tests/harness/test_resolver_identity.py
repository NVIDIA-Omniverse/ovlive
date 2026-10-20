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
from tools.live.resolver_identity import inputs, SDK_TREES
from tools.live.test_support import ProcessResult

SCRIPT = Path(__file__).resolve().parents[2] / "tools/live/resolver_identity.py"


class ResolverIdentityTests(unittest.TestCase):
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
        self.artifact = self.root / "resolver-driver"
        self.artifact.write_bytes(b"synthetic driver; never executed")
        self.plugin = self.root / "resolver-plugin.so"
        self.plugin.write_bytes(b"synthetic test plugin")
        self.plug_info = self.root / "plugInfo.json"
        self.plug_info.write_text(json.dumps({"Plugins": [{"LibraryPath": str(self.plugin)}]}))
        for name in ("Probe.cpp", "Plugin.cpp", "Probe.h", "plugInfo.json.in", "README.md", "creation.py"):
            path = self.root / "tests/native/resolver" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic probe " + name)
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
            *[item for name, path in self.sdks.items() for item in ("--" + name, str(path))],
            "--compiler-id", "Synthetic", "--compiler-version", "1", "--configuration", "Debug",
            "--sanitizers", "1"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.manifest_path = Path(str(self.artifact) + ".manifest.json")
        self.manifest = json.loads(self.manifest_path.read_text())

    def test_source_driver_plugin_and_generated_metadata_are_checked(self):
        self.assertEqual(runner.native_identity(self.artifact, "resolver_probe"), self.manifest)
        for name in inputs(self.root):
            with self.subTest(source=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                with self.assertRaisesRegex(ValueError, "stale source"):
                    runner.native_identity(self.artifact, "resolver_probe")
                path.write_bytes(original)
        for path in (self.plugin, self.plug_info):
            with self.subTest(dependency=path):
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                with self.assertRaisesRegex(ValueError, "stale dependency"):
                    runner.native_identity(self.artifact, "resolver_probe")
                path.write_bytes(original)
        self.artifact.write_bytes(b"changed binary")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            runner.native_identity(self.artifact, "resolver_probe")

    def test_all_sdk_trees_and_missing_directory_are_checked(self):
        for entry in self.manifest["dependency_trees"]:
            with self.subTest(role=entry["role"]):
                path = Path(entry["path"]) / "input"
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                with self.assertRaisesRegex(ValueError, "SDK differs"):
                    runner.native_identity(self.artifact, "resolver_probe")
                path.unlink()
                path.parent.rmdir()
                with self.assertRaisesRegex(ValueError, "SDK directory is missing"):
                    runner.native_identity(self.artifact, "resolver_probe")
                path.parent.mkdir()
                path.write_bytes(original)

    def test_missing_roles_wrong_boundary_and_unrecorded_plugin_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unexpected native artifact"):
            runner.native_identity(self.artifact, "sdf_scene")
        for field, phrase in (("dependencies", "plugin/metadata identity"), ("dependency_trees", "SDK identity")):
            changed = dict(self.manifest)
            changed[field] = self.manifest[field][:-1]
            self.manifest_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, phrase):
                runner.native_identity(self.artifact, "resolver_probe")
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.plug_info.write_text(json.dumps({"Plugins": [{"LibraryPath": "different-unrecorded-plugin.so"}]}))
        next(entry for entry in self.manifest["dependencies"] if entry["role"] == "plug_info")["sha256"] = \
            hashlib.sha256(self.plug_info.read_bytes()).hexdigest()
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "unrecorded binary"):
            runner.native_identity(self.artifact, "resolver_probe")

    def run_runner(self, arguments, process):
        artifacts = self.root / "runs"
        before = set(artifacts.iterdir()) if artifacts.exists() else set()
        # Mocked workloads exercise identity, completion and replay for admitted suites.
        with patch.object(sys, "argv", ["test.py", *arguments, "--artifact-dir", str(artifacts)]), \
                patch.object(runner, "suite_platform_error", return_value=None), \
                patch.object(runner, "fingerprint", return_value="source-fingerprint"), \
                patch.object(runner, "identity", return_value="toolchain"), \
                patch.object(runner, "run_process", side_effect=process), contextlib.redirect_stdout(io.StringIO()):
            status = runner.main()
        return status, next(iter(set(artifacts.iterdir()) - before))

    def test_runner_missing_stale_boundary_replay_and_per_run_trace(self):
        executed = []

        def process(command, **options):
            executed.append((command, options["environment"].copy()))
            self.write_probe_traces(command)
            return ProcessResult("passed", 0, 1, 0, False, True)

        status, missing = self.run_runner(["--suite", "resolver-probe"], process)
        self.assertEqual(status, 2)
        self.assertIn("--resolver-driver", json.loads((missing / "manifest.json").read_text())["results"][0]["reason"])
        args = ["--suite", "resolver-probe", "--seed", "42", "--resolver-driver", str(self.artifact)]
        self.manifest_path.write_text(json.dumps(dict(self.manifest, boundary="sdf_scene")))
        status, rejected = self.run_runner(args, process)
        self.assertEqual(status, 2)
        self.assertIn("unexpected native artifact", json.loads((rejected / "manifest.json").read_text())["results"][0]["reason"])
        self.manifest_path.write_text(json.dumps(self.manifest))
        source = self.root / "tests/native/resolver/Probe.cpp"
        original = source.read_bytes()
        source.write_bytes(original + b" stale")
        status, rejected = self.run_runner(args, process)
        self.assertEqual(status, 2)
        self.assertIn("stale source", json.loads((rejected / "manifest.json").read_text())["results"][0]["reason"])
        self.assertEqual(executed, [])
        source.write_bytes(original)
        status, first = self.run_runner(args, process)
        self.assertEqual(status, 0)
        replay = json.loads((first / "replay.json").read_text())
        self.assertEqual(replay["resolver_identity"], self.manifest)
        self.assertEqual(executed[0][0], [sys.executable, "-B", str(self.root / "tests/native/resolver/creation.py"),
                         str(self.artifact), str(self.plug_info), str(first / "resolver-probe"), "--with-baseline"])
        self.assertEqual(executed[0][1]["LIVE_TEST_SEED"], "42")
        self.assertEqual(executed[0][1]["ASAN_OPTIONS"], sanitizer.runtime_options()["ASAN_OPTIONS"])
        self.assertEqual(executed[0][1]["LSAN_OPTIONS"], sanitizer.runtime_options()["LSAN_OPTIONS"])
        self.assertEqual(executed[0][1]["UBSAN_OPTIONS"], "halt_on_error=1:print_stacktrace=1")
        for key in ("PXR_PLUGINPATH_NAME", "PXR_AR_DEFAULT_SEARCH_PATH", "AR_DEFAULT_RESOLVER"):
            self.assertEqual(executed[0][1][key], "")
        status, second = self.run_runner(["--replay", str(first / "replay.json")], process)
        self.assertEqual(status, 0)
        self.assertNotEqual(executed[0][0][-2], executed[1][0][-2])
        self.assertTrue((first / "resolver-probe/trace.json").is_file())
        self.assertTrue((second / "resolver-probe/trace.json").is_file())
        self.assertTrue((first / "resolver-probe/creation/summary.json").is_file())
        self.manifest["compiler_version"] = "new build"
        self.manifest_path.write_text(json.dumps(self.manifest))
        status, rejected = self.run_runner(["--replay", str(first / "replay.json")], process)
        self.assertEqual(status, 2)
        self.assertEqual(len(executed), 2)
        self.assertIn("artifact differs from replay", json.loads((rejected / "manifest.json").read_text())["results"][0]["reason"])

    def test_success_without_trace_is_an_infrastructure_failure(self):
        status, directory = self.run_runner(
            ["--suite", "resolver-probe", "--resolver-driver", str(self.artifact)],
            lambda *args, **kwargs: ProcessResult("passed", 0, 1, 0, False, True))
        self.assertEqual(status, 2)
        result = json.loads((directory / "manifest.json").read_text())["results"][0]
        self.assertEqual(result["outcome"], "infrastructure_error")
        self.assertIn("resolver trace missing or invalid", result["reason"])

    def test_plugin_environment_changes_are_confined_to_probe(self):
        environments = []

        def process(command, **options):
            environments.append(options["environment"].copy())
            if command[-1] == "--with-baseline":
                self.write_probe_traces(command)
            else:
                directory = Path(options["environment"]["LIVE_TEST_ARTIFACT_DIR"])
                (directory / "foundation.json").write_text(json.dumps({
                    "schema_version": 1, "boundary": "foundation", "passed": True,
                    "discovered": 1, "tests_run": 1, "skipped": 0, "expected_failures": 0,
                    "test_ids": ["synthetic.environment.fixture"],
                    "excluded": ([dict(id=name, reason=reason) for name, reason in runner.MACOS_EXCLUSIONS.items()]
                                 if sys.platform == 'darwin' else []),
                    "cases": [{"id": "synthetic.environment.fixture", "outcome": "passed", "message": ""}]}))
            return ProcessResult("passed", 0, 1, 0, False, True)

        status, _ = self.run_runner(["--suite", "resolver-probe,foundation", "--resolver-driver", str(self.artifact)], process)
        self.assertEqual(status, 0)
        for key in ("PXR_PLUGINPATH_NAME", "PXR_AR_DEFAULT_SEARCH_PATH", "AR_DEFAULT_RESOLVER"):
            self.assertEqual(environments[0][key], "")
            self.assertNotIn(key, environments[1])

    @staticmethod
    def write_probe_traces(command):
        directory = Path(command[-2])
        (directory / 'trace.json').write_text(json.dumps({"schema_version": 1, "boundary": "usd_resolver_hook_probe",
            "usd_version": 2511, "passed": True, "steps": [{"name": "synthetic", "events": []}]}))
        (directory / 'creation').mkdir()
        ordered = {
            'create-deferred-work': ['queued-anonymous-work', 'outer-create-returned', 'before-deferred-anonymous', 'after-deferred-anonymous'],
            'create-deferred-save-notice': ['save-notice-queued', 'queued-anonymous-work', 'outer-create-returned', 'before-deferred-anonymous', 'after-deferred-anonymous'],
            'create-release-failure': ['outer-create-returned', 'failed-target-destroyed', 'before-find-after-failure', 'after-find-after-failure'],
            'create-freeze-failure': ['after-initial-data-swap', 'outer-create-returned', 'failed-target-destroyed', 'frozen-backing-survived-target', 'before-find-after-failure', 'after-find-after-failure'],
            'create-install': ['before-initial-data-swap', 'after-initial-data-swap', 'outer-create-returned'],
            'export-nested': ['before-nested-anonymous', 'after-nested-anonymous'],
        }
        contract = dict(schema_version=1, boundary='usd_creation_hook_contract', usd_version=2511, passed=True,
                        service=False, production_plugin=False,
                        cases=[dict(name=name, passed=True, timed_out=False, returncode=0, output_limited=False,
                                    markers=['anonymous-control-passed', 'content-notice-control-passed', *markers, 'completed'],
                                    hooks=['WriteToFile' if name == 'export-nested' else 'SaveToFile'])
                               for name, markers in ordered.items()])
        path = directory / 'creation/summary.json'
        path.write_text(json.dumps(contract))
        return path

    def test_missing_or_incomplete_creation_contract_cannot_report_probe_success(self):
        for mutation in ('missing', 'case', 'timeout', 'unexpected-return', 'markers', 'order', 'hook', 'production', 'output-limit'):
            def process(command, **options):
                path = self.write_probe_traces(command)
                if mutation == 'missing':
                    path.unlink()
                else:
                    contract = json.loads(path.read_text())
                    if mutation == 'case': contract['cases'].pop()
                    if mutation == 'timeout': contract['cases'][0]['timed_out'] = True
                    if mutation == 'unexpected-return': contract['cases'][0]['returncode'] = -9
                    if mutation == 'markers': contract['cases'][0]['markers'] = []
                    if mutation == 'order': contract['cases'][0]['markers'].reverse()
                    if mutation == 'hook': contract['cases'][-1]['hooks'] = ['SaveToFile']
                    if mutation == 'production': contract['production_plugin'] = True
                    if mutation == 'output-limit': contract['cases'][0]['output_limited'] = True
                    path.write_text(json.dumps(contract))
                return ProcessResult("passed", 0, 1, 0, False, True)
            with self.subTest(mutation=mutation):
                status, directory = self.run_runner(
                    ["--suite", "resolver-probe", "--resolver-driver", str(self.artifact)], process)
                self.assertEqual(status, 2)
                self.assertEqual(json.loads((directory / 'manifest.json').read_text())['results'][0]['outcome'],
                                 'infrastructure_error')


if __name__ == "__main__":
    unittest.main()
