# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kit runner evidence must bind complete cases, child retirement and selected inputs."""
import json
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.live import kit_reference as kit, test as runner
from tools.live.test_support import ProcessResult

SUFFIX = ".cpython-312-x86_64-linux-gnu.so"
# The adapter identity a snapshot records and the loaded module and interpreter ABI a Kit probe reports for it.
ADAPTER = dict(path="/synthetic/omni_usd_live/python/omni/usd/live/_live" + SUFFIX, sha256="a" * 64, ext_suffix=SUFFIX)
LOADED = dict(python_abi=dict(soabi=SUFFIX[1:-3], ext_suffix=SUFFIX, cache_tag="cpython-312"),
              live_module=dict(path=ADAPTER["path"], sha256=ADAPTER["sha256"], abi_version=0x00010000))


class KitReferenceRunnerTests(unittest.TestCase):
    def test_common_runner_replay_requires_runtime_and_detects_changed_valid_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "synthetic.so"
            library.write_bytes(b"runtime")
            paths = {name: str(root / name) for name in kit.INPUTS}
            identity = dict(lane=kit.BUILD_TREE, paths=paths, executable=dict(sha256="executable"),
                            extension_configs={"synthetic": "config"}, adapter=ADAPTER,
                            required_libraries={str(library): kit.digest(library)},
                            candidate=dict(plugin_identity=dict(dependency_trees=[])))
            process = dict(outcome="passed", returncode=0, cleanup_complete=True, output_truncated=False)
            probe = dict(passed=True, stage_root="live", python_version="synthetic-python", python_optimize=0, dont_write_bytecode=True,
                         settings=kit.SETTINGS, extension_configs=identity["extension_configs"],
                         executable_sha256="executable", driver_sha256=kit.digest(kit.ROOT / "tests/native/python/kit_process.py"),
                         native_libraries=identity["required_libraries"], **LOADED)
            execution = {"passed": True}

            def execute(command, **options):
                directory = Path(options["environment"]["LIVE_TEST_ARTIFACT_DIR"])
                for test, count in kit.TESTS.items():
                    for index in range(count):
                        child = directory / (test.rsplit(".", 1)[-1] + "-kit-" + str(index))
                        child.mkdir()
                        (child / "process.json").write_text(json.dumps(process))
                        (child / "probe.json").write_text(json.dumps(dict(probe, stage_root='live' if child.name == 'test_kit_direct_live_root_issues_host_edits-kit-0' else 'composed')))
                summary = dict(schema_version=1, boundary="kit-reference", profile=kit.PROFILE,
                               passed=execution["passed"], test_ids=sorted(kit.TESTS), discovered=len(kit.TESTS), tests_run=len(kit.TESTS),
                               cases=[dict(id=test, outcome="passed" if execution["passed"] else "failed", message="")
                                      for test in sorted(kit.TESTS)],
                               skipped=0, failures=0 if execution["passed"] else 1,
                               errors=0, expected_failures=0, unexpected_successes=0,
                               seed=42, source_fingerprint="source", final_source_fingerprint="source",
                               inputs_unchanged=True, evidence_errors=[], kit_identity=identity,
                               kit_lane=kit.BUILD_TREE, release_package_sha256=None,
                               participants=kit.participants(directory, identity))
                (directory / "kit-reference-result.json").write_text(json.dumps(summary))
                return ProcessResult("passed" if execution["passed"] else "failed",
                                     0 if execution["passed"] else 1, 0, 0, False, True)

            def run(arguments):
                # Mocked workloads exercise completion and replay for an admitted suite.
                with patch.object(sys, "argv", ["test.py", *arguments]), \
                        patch.object(runner, "suite_platform_error", return_value=None), \
                        patch.object(runner, "fingerprint", return_value="source"), \
                        patch.object(kit, "snapshot", return_value=identity), \
                        patch.object(runner.platform, "platform", return_value="synthetic-platform"), \
                        patch.object(runner.subprocess, "run", return_value=SimpleNamespace(stdout="{}")), \
                        patch.object(runner, "run_process", side_effect=execute) as child, \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        code = runner.main()
                    except SystemExit as error:
                        code = error.code
                    return code, child.call_count

            fresh = ["--suite", "kit-reference", "--seed", "42", "--artifact-dir", str(root / "fresh"),
                     "--reference-python", sys.executable, "--reference-openssl", sys.executable]
            for name, path in paths.items():
                fresh += ["--" + name.replace("_", "-"), path]
            self.assertEqual(run(fresh), (0, 1))
            record = json.loads((next((root / "fresh").iterdir()) / "replay.json").read_text())
            execution["passed"] = False
            self.assertEqual(run([str(root / "failed") if arg == str(root / "fresh") else arg for arg in fresh]), (1, 1))
            failed = json.loads((next((root / "failed").iterdir()) / "replay.json").read_text())
            self.assertEqual(failed["kit_runtime"], record["kit_runtime"])
            execution["passed"] = True
            recipe = root / "replay.json"
            arguments = ["--replay", str(recipe), "--artifact-dir", str(root / "replayed")]
            recipe.write_text(json.dumps(record))
            self.assertEqual(run(arguments), (0, 1))
            for value in (None, [], {}):
                recipe.write_text(json.dumps(dict(record, kit_runtime=value)))
                with self.subTest(runtime=value):
                    self.assertEqual(run(arguments), (2, 0))
            missing = dict(record)
            missing.pop("kit_runtime")
            recipe.write_text(json.dumps(missing))
            self.assertEqual(run(arguments), (2, 0))
            changed = json.loads(json.dumps(record))
            next(iter(changed["kit_runtime"].values()))["python_version"] = "different-valid-runtime"
            recipe.write_text(json.dumps(changed))
            self.assertEqual(run(arguments), (2, 1))
            recipe.write_text(json.dumps(record))
            library.write_bytes(b"changed library")
            self.assertEqual(run(arguments), (2, 0))

    def test_tree_follows_overlay_links_but_never_reads_package_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            overlay, external = root / "overlay", root / "external"
            overlay.mkdir(); external.mkdir()
            (external / "module.py").write_text("original")
            (overlay / "python").symlink_to(external, target_is_directory=True)
            metadata = external / "PACKAGE-INFO.yaml"
            metadata.write_text("must not be read")
            original = kit.digest
            def checked(path):
                self.assertNotEqual(Path(path).name, "PACKAGE-INFO.yaml")
                return original(path)
            with patch.object(kit, "digest", side_effect=checked):
                before = kit.tree(overlay)
                self.assertEqual(before["files"], 1)
                metadata.write_text("changed excluded metadata")
                self.assertEqual(kit.tree(overlay), before)
                (external / "metadata-alias").symlink_to(metadata)
                self.assertEqual(kit.tree(overlay), before)
                (external / "module.py").write_text("changed")
                self.assertNotEqual(kit.tree(overlay), before)
                (external / "loop").symlink_to(overlay, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "cycle"):
                    kit.tree(overlay)

    def test_selected_overlay_changes_and_missing_inputs_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(**{name: None for name in kit.INPUTS})
            if not sys.platform.startswith("linux"):
                # Kit reference qualification is Linux-only; elsewhere it refuses before reading any input.
                with self.assertRaisesRegex(ValueError, "requires Linux"):
                    kit.snapshot(args)
                return
            with self.assertRaisesRegex(ValueError, "kit-executable"):
                kit.snapshot(args)
            args.kit_executable = root / "kit"
            args.kit_executable.write_bytes(b"synthetic executable")
            args.kit_executable.chmod(0o700)
            args.kit_package = root / "package"
            module = args.kit_package / "python/omni/usd/live/_live.so"
            module.parent.mkdir(parents=True); module.write_bytes(b"synthetic adapter")
            for name in ("kit_client_extension", "kit_resolver_extension"):
                path = root / name
                (path / "config").mkdir(parents=True)
                (path / "config/extension.toml").write_text("synthetic config")
                setattr(args, name, path)
            (args.kit_package / "config").mkdir()
            (args.kit_package / "config/extension.toml").write_text("synthetic package config")
            for name, relative in (("kit_package", "lib/libomni_usd_live.so"),
                                   ("kit_client_extension", "bin/libomni.client.lifecycle.plugin.so"),
                                   ("kit_resolver_extension", "lib/libcarb.usdresolver.plugin.so"),
                                   ("kit_resolver_extension", "lib/libomni_usd_resolver.so"),
                                   ("kit_resolver_extension", "bin/libomni.ext-usd_resolver.plugin.so")):
                path = getattr(args, name) / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic library")
            with patch("tools.live.python_identity.validate", return_value={"synthetic": "candidate"}):
                before = kit.snapshot(args)
                (args.kit_resolver_extension / "config/extension.toml").write_text("changed")
                self.assertNotEqual(kit.snapshot(args), before)
                args.kit_core_extension = root / 'core'
                config = args.kit_core_extension / 'config/extension.toml'
                config.parent.mkdir(parents=True)
                config.write_text('selected core')
                tbb = args.kit_core_extension / 'bin/libomni.tbb.globalcontrol.plugin.so'
                tbb.parent.mkdir()
                tbb.symlink_to(library := root / 'companion-tbb.so')
                library.write_bytes(b'companion')
                selected = kit.snapshot(args)
                self.assertEqual(selected['required_libraries'][str(library)], kit.digest(library))
                self.assertEqual(selected['extension_configs']['omni.usd.core'], kit.digest(config))
                self.assertIn('--kit-core-extension', kit.arguments(selected))
                library.write_bytes(b'changed companion')
                self.assertNotEqual(kit.snapshot(args), selected)

    def test_completion_requires_every_case_child_and_unchanged_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "synthetic-runtime.so"
            library.write_bytes(b"synthetic runtime")
            configs = {key: "synthetic-config" for key in ("omni_usd_live", "omni.client", "omni.usd_resolver")}
            identity = dict(lane=kit.BUILD_TREE, executable=dict(sha256="synthetic-executable"),
                            extension_configs=configs, adapter=ADAPTER,
                            required_libraries={str(library): kit.digest(library)})
            process = dict(outcome="passed", returncode=0, cleanup_complete=True, output_truncated=False)
            probe = dict(passed=True, stage_root="live", python_version="synthetic-python", python_optimize=0,
                         dont_write_bytecode=True, settings=kit.SETTINGS, extension_configs=configs,
                         driver_sha256=kit.digest(kit.ROOT / "tests/native/python/kit_process.py"),
                         executable_sha256="synthetic-executable",
                         native_libraries={str(library): kit.digest(library)}, **LOADED)
            children = []
            for name, count in kit.TESTS.items():
                for index in range(count):
                    child = root / (name.rsplit(".", 1)[-1] + "-kit-" + str(index))
                    child.mkdir(); children.append(child)
                    (child / "process.json").write_text(json.dumps(process))
                    (child / "probe.json").write_text(json.dumps(dict(probe, stage_root='live' if child.name == 'test_kit_direct_live_root_issues_host_edits-kit-0' else 'composed')))
            runtime = kit.participants(root, identity)
            summary = dict(schema_version=1, boundary="kit-reference", profile=kit.PROFILE,
                           passed=True, test_ids=sorted(kit.TESTS), discovered=len(kit.TESTS), tests_run=len(kit.TESTS),
                           skipped=0, failures=0, errors=0, expected_failures=0, unexpected_successes=0,
                           seed=42, source_fingerprint="source", final_source_fingerprint="source",
                           inputs_unchanged=True, evidence_errors=[], kit_identity=identity,
                           kit_lane=kit.BUILD_TREE, release_package_sha256=None, participants=runtime)
            path = root / "kit-reference-result.json"
            path.write_text(json.dumps(summary))
            self.assertEqual(kit.completion(root, identity, "source", 42), summary)
            kit.validate_runtime(runtime, identity)
            for change in (dict(discovered=0), dict(tests_run=1), dict(skipped=1), dict(test_ids=[]),
                           dict(expected_failures=1), dict(inputs_unchanged=False), dict(seed=True),
                           dict(final_source_fingerprint="changed"), dict(participants={}),
                           dict(evidence_errors=["participant_evidence_incomplete_or_changed"]),
                           dict(kit_lane=kit.RELEASE_PACKAGE), dict(release_package_sha256="0" * 64)):
                path.write_text(json.dumps(dict(summary, **change)))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    kit.completion(root, identity, "source", 42)
            path.write_text(json.dumps(summary))
            for change in (dict(settings={}), dict(extension_configs={}), dict(python_optimize=1),
                           dict(python_optimize=False), dict(dont_write_bytecode=False), dict(native_libraries={}), dict(stage_root="composed"),
                           dict(live_module=dict(LOADED["live_module"], path="/other/_live.so")),
                           dict(live_module=dict(LOADED["live_module"], sha256="0" * 64)),
                           dict(live_module=dict(LOADED["live_module"], abi_version="0x00010000")),
                           dict(python_abi=dict(LOADED["python_abi"], ext_suffix=".cpython-311-x86_64-linux-gnu.so"))):
                (children[0] / "probe.json").write_text(json.dumps(dict(probe, **change)))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    kit.completion(root, identity, "source", 42)
            (children[0] / "probe.json").write_text(json.dumps(probe))
            extra = root / "test_unexpected-kit-0"
            extra.mkdir()
            with self.assertRaises(ValueError):
                kit.completion(root, identity, "source", 42)
            extra.rmdir()
            (children[0] / "process.json").write_text(json.dumps(dict(process, cleanup_complete=False)))
            with self.assertRaises(ValueError):
                kit.completion(root, identity, "source", 42)
            (children[0] / "process.json").write_text(json.dumps(process))
            library.write_bytes(b"changed runtime")
            with self.assertRaises(ValueError):
                kit.completion(root, identity, "source", 42)
            with self.assertRaises(ValueError):
                kit.validate_runtime(runtime, identity)

    def test_common_runner_cannot_accept_zero_exit_without_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {name: str(root / name) for name in kit.INPUTS}
            identity = dict(lane=kit.BUILD_TREE, paths=paths, candidate=dict(plugin_identity=dict(dependency_trees=[
                dict(role="usd_lib", path=str(root / "usd/lib")),
                dict(role="python_lib", path=str(root / "python/lib"))])))
            argv = ["test.py", "--suite", "kit-reference", "--seed", "42", "--artifact-dir", str(root / "out"),
                    "--reference-python", sys.executable, "--reference-openssl", sys.executable]
            for name, path in paths.items():
                argv += ["--" + name.replace("_", "-"), path]
            result = ProcessResult("passed", 0, 0, 0, False, True)
            # The admitted mocked workload must still supply complete Kit evidence.
            with patch.object(sys, "argv", argv), patch.object(runner, "fingerprint", return_value="source"), \
                    patch.object(runner, "suite_platform_error", return_value=None), \
                    patch.object(kit, "snapshot", return_value=identity), \
                    patch.object(runner.platform, "platform", return_value="synthetic-platform"), \
                    patch.object(runner.subprocess, "run", return_value=SimpleNamespace(stdout="{}")), \
                    patch.object(runner, "run_process", return_value=result) as child, \
                    patch.dict(os.environ, {"SERVICE_SECRET": "synthetic-secret"}):
                self.assertEqual(runner.main(), 2)
            command = child.call_args.args[0]
            for name, path in paths.items():
                self.assertEqual(command[command.index("--" + name.replace("_", "-")) + 1], path)
            self.assertNotIn("SERVICE_SECRET", child.call_args.kwargs["environment"])
            directory = next((root / "out").iterdir())
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertFalse(manifest["passed"])
            self.assertIn("Invalid Kit completion", manifest["results"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
