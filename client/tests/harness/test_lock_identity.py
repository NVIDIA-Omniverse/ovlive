# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from tools.live import sanitizer
from tools.live import test as runner
from tools.live.test_support import ProcessResult


class LockIdentityTests(unittest.TestCase):
    def setUp(self):
        # ROOT below is a synthetic source identity fixture. Keep the real
        # replay JSON parser importable when this module runs in isolation.
        search_path = patch.object(sys, "path", [str(runner.ROOT / "tests/reference_service"), *sys.path])
        search_path.start()
        self.addCleanup(search_path.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifact = self.root / "driver"
        self.artifact.write_bytes(b"synthetic driver")
        names = ("CMakeLists.txt", "cmake/LiveLockContract.cmake",
                 "source/live/support/ReaderWriterLock.h",
                 "source/live/support/ReaderWriterLock.inl",
                 "tests/native/core/lock_noalloc.cpp", "tools/live/write_build_manifest.py",
                 "cmake/LiveTestSupport.cmake", "tests/native/support/test_tls.h")
        self.inputs = []
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic " + name)
            self.inputs.append(dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        self.manifest = dict(schema_version=1, boundary="lock_noalloc", sanitizers=True,
                             artifact_sha256=hashlib.sha256(self.artifact.read_bytes()).hexdigest(),
                             inputs=self.inputs, dependencies=[])
        self.manifest_path = Path(str(self.artifact) + ".manifest.json")
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.patch = patch.object(runner, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_source_and_artifact_identity(self):
        self.assertEqual(runner.native_identity(self.artifact, "lock_noalloc"), self.manifest)
        for entry in self.inputs:
            path = self.root / entry["path"]
            original = path.read_bytes()
            path.write_bytes(original + b" changed")
            with self.assertRaisesRegex(ValueError, "stale source"):
                runner.native_identity(self.artifact, "lock_noalloc")
            path.write_bytes(original)
        missing = dict(self.manifest, inputs=self.inputs[:-1])
        self.manifest_path.write_text(json.dumps(missing))
        with self.assertRaisesRegex(ValueError, "missing required source"):
            runner.native_identity(self.artifact, "lock_noalloc")
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.artifact.write_bytes(b"different driver")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            runner.native_identity(self.artifact, "lock_noalloc")

    def test_runner_seed_sanitizers_missing_driver_and_replay(self):
        artifacts = self.root / "runs"
        commands = []
        def process(command, **options):
            commands.append((command, options["environment"]))
            options["output"].write_text("live-cases passed=2\n")
            return ProcessResult("passed", 0, 1, 0, False, True)
        def run(arguments):
            before = set(artifacts.iterdir()) if artifacts.exists() else set()
            with patch.object(sys, "argv", ["test.py", *arguments, "--artifact-dir", str(artifacts)]), \
                    patch.object(runner, "fingerprint", return_value="fixed-source"), \
                    patch.object(runner, "identity", return_value="toolchain"), \
                    patch.object(runner, "run_process", side_effect=process), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = runner.main()
            return status, next(iter(set(artifacts.iterdir()) - before))
        self.assertEqual(run(["--suite", "lock-noalloc"])[0], 2)
        self.assertEqual(commands, [])
        status, first = run(["--suite", "lock-noalloc", "--seed", "31415", "--lock-driver", str(self.artifact)])
        self.assertEqual(status, 0)
        self.assertEqual(commands[0][0], [str(self.artifact)])
        self.assertEqual(commands[0][1]["LIVE_TEST_SEED"], "31415")
        self.assertEqual(commands[0][1]["ASAN_OPTIONS"], sanitizer.runtime_options()["ASAN_OPTIONS"])
        self.assertEqual(commands[0][1]["LSAN_OPTIONS"], sanitizer.runtime_options()["LSAN_OPTIONS"])
        replay = json.loads((first / "replay.json").read_text())
        self.assertEqual(replay["lock_identity"], self.manifest)
        self.assertEqual(run(["--replay", str(first / "replay.json")])[0], 0)
        self.assertEqual(commands[1][0], commands[0][0])
        altered = self.root / "altered-replay.json"
        for key in ("lock_driver", "lock_identity"):
            for value in (None, "", {}):
                with self.subTest(key=key, value=value):
                    altered.write_text(json.dumps(dict(replay, **{key: value})))
                    with self.assertRaises(SystemExit) as error:
                        run(["--replay", str(altered), "--lock-driver", str(self.artifact)])
                    self.assertEqual(error.exception.code, 2)
            missing = dict(replay)
            del missing[key]
            altered.write_text(json.dumps(missing))
            with self.assertRaises(SystemExit) as error:
                run(["--replay", str(altered), "--lock-driver", str(self.artifact)])
            self.assertEqual(error.exception.code, 2)
        # A newly valid artifact is still the wrong candidate for this replay.
        alternate = self.root / "alternate-driver"
        alternate.write_bytes(b"different but valid candidate")
        alternate_manifest = dict(self.manifest, sanitizers=False,
                                  artifact_sha256=hashlib.sha256(alternate.read_bytes()).hexdigest())
        Path(str(alternate) + ".manifest.json").write_text(json.dumps(alternate_manifest))
        self.assertEqual(runner.native_identity(alternate, "lock_noalloc"), alternate_manifest)
        altered.write_text(json.dumps(dict(replay, lock_driver=str(alternate))))
        self.assertEqual(run(["--replay", str(altered)])[0], 2)
        self.assertEqual(len(commands), 2)
        self.artifact.write_bytes(b"stale")
        self.assertEqual(run(["--replay", str(first / "replay.json")])[0], 2)
        self.assertEqual(len(commands), 2)

    def test_runner_requires_the_driver_case_count(self):
        artifacts = self.root / "case-count-runs"
        for log in ('', 'live-cases passed=1\n', 'live-cases passed=2\nlive-cases passed=2\n'):
            with self.subTest(log=log):
                def process(command, **options):
                    options["output"].write_text(log)
                    return ProcessResult("passed", 0, 1, 0, False, True)
                before = set(artifacts.iterdir()) if artifacts.exists() else set()
                with patch.object(sys, "argv", ["test.py", "--suite", "lock-noalloc", "--seed", "7",
                                                "--lock-driver", str(self.artifact), "--artifact-dir", str(artifacts)]), \
                        patch.object(runner, "fingerprint", return_value="synthetic-source"), \
                        patch.object(runner, "identity", return_value="synthetic-toolchain"), \
                        patch.object(runner, "run_process", side_effect=process), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(), 2)
                run = next(iter(set(artifacts.iterdir()) - before))
                result = json.loads((run / "manifest.json").read_text())["results"][0]
                self.assertEqual(result["outcome"], "infrastructure_error")
                self.assertIn("live-cases passed=2", result["reason"])
