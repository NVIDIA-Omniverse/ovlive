# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selection by tag, package, scenario, shard and driver mode; fingerprints, budgets, redaction and interrupts."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.live import test as runner
from tools.live.test_support import ProcessResult, in_shard, parse_shard, redact, select_cases

ROOT = Path(__file__).resolve().parents[2]


def synthetic_suite(ids):
    """A suite whose case IDs are exactly `ids`."""
    class Case(unittest.TestCase):
        def __init__(self, identity):
            super().__init__("runTest")
            self.identity = identity

        def id(self):
            return self.identity

        def runTest(self):
            pass
    return unittest.TestSuite(Case(identity) for identity in ids)


def foundation_process(record=None, log="", extra=None):
    """A run_process stand-in that writes complete foundation evidence and records its options."""
    def process(command, **options):
        if record is not None:
            record.append(options)
        directory = Path(options["environment"]["LIVE_TEST_ARTIFACT_DIR"])
        ids = ["test_a.Tests.test_one"]
        summary = dict(schema_version=1, boundary="foundation", discovered=1, tests_run=1, skipped=0,
                       expected_failures=0, passed=True, test_ids=ids,
                       excluded=([dict(id=name, reason=reason) for name, reason in runner.MACOS_EXCLUSIONS.items()]
                                 if sys.platform == 'darwin' else []),
                       cases=[dict(id=ids[0], outcome="passed", message="")])
        (directory / "foundation.json").write_text(json.dumps(summary))
        options["output"].write_text(log)
        if extra:
            extra(directory)
        return ProcessResult("passed", 0, 1, len(log), False, True)
    return process


class CaseSelectionTests(unittest.TestCase):
    IDS = [f"test_m{module}.Tests.test_{case}" for module in range(4) for case in range(25)]

    def test_shards_partition_cases_by_id(self):
        for count in (1, 2, 3, 7):
            with self.subTest(count=count):
                shards = [[case.id() for case in select_cases(synthetic_suite(self.IDS), shard=(index, count))]
                          if any(in_shard(i, (index, count)) for i in self.IDS) else []
                          for index in range(count)]
                self.assertEqual(sorted(sum(shards, [])), sorted(self.IDS))
                # A case's shard depends on its ID alone, not on the other cases or their order.
                reversed_ids = list(reversed(self.IDS))
                for index in range(count):
                    if shards[index]:
                        chosen = [case.id() for case in select_cases(synthetic_suite(reversed_ids), shard=(index, count))]
                        self.assertEqual(sorted(chosen), sorted(shards[index]))

    def test_scenarios_match_exact_ids_or_dotted_prefixes(self):
        chosen = select_cases(synthetic_suite(self.IDS), ["test_m1.Tests", "test_m2.Tests.test_3", "test_m3"])
        self.assertEqual([case.id() for case in chosen],
                         [i for i in self.IDS if i.startswith(("test_m1.", "test_m3.")) or i == "test_m2.Tests.test_3"])
        # `test_m2.Tests.test_1` is not a prefix match of `test_m2.Tests.test_10`.
        self.assertEqual([case.id() for case in select_cases(synthetic_suite(self.IDS), ["test_m2.Tests.test_1"])],
                         ["test_m2.Tests.test_1"])

    def test_a_selection_that_keeps_nothing_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "scenario selects no case: test_missing"):
            select_cases(synthetic_suite(self.IDS), ["test_m0", "test_missing"])
        empty = next(index for index in range(50) if not any(in_shard(i, (index, 50)) for i in self.IDS[:1]))
        with self.assertRaisesRegex(ValueError, "keeps no case"):
            select_cases(synthetic_suite(self.IDS[:1]), shard=(empty, 50))

    def test_a_module_that_failed_to_import_stays_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "test_broken.py").write_text("raise ImportError('synthetic')\n")
            (Path(temporary) / "test_fine.py").write_text(
                "import unittest\nclass Tests(unittest.TestCase):\n    def test_ok(self):\n        pass\n")
            suite = unittest.defaultTestLoader.discover(temporary, top_level_dir=temporary)
            chosen = [case.id() for case in select_cases(suite, ["test_fine"])]
        self.assertEqual(sorted(chosen), ["test_fine.Tests.test_ok", "unittest.loader._FailedTest.test_broken"])

    def test_shard_text_is_strict(self):
        self.assertEqual(parse_shard("1/3"), (1, 3))
        for text in ("3/3", "-1/3", "1/0", "01/3", "1/3/5", "a/b", "1"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_shard(text)


class RunnerSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.artifacts = Path(temporary.name) / "runs"

    def run_main(self, arguments, process=None, catalog=None):
        before = set(self.artifacts.iterdir()) if self.artifacts.exists() else set()
        with patch.object(sys, "argv", ["test.py", *arguments, "--artifact-dir", str(self.artifacts)]), \
                patch.object(runner, "fingerprint", return_value="synthetic-source"), \
                patch.object(runner, "identity", return_value="synthetic-toolchain"), \
                patch.object(runner, "run_process", side_effect=process), \
                patch.object(runner, "CATALOG", catalog or runner.CATALOG), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = runner.main()
        run = next(iter(set(self.artifacts.iterdir()) - before))
        return status, run, json.loads((run / "manifest.json").read_text())

    def rejected(self, arguments):
        with patch.object(sys, "argv", ["test.py", *arguments, "--artifact-dir", str(self.artifacts)]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as error:
            runner.main()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.artifacts.exists() and any(self.artifacts.iterdir()), "a rejected selection ran")

    def test_tags_and_packages_select_implemented_suites_in_catalog_order(self):
        self.assertEqual(runner.select_suites(None, ["pr"], []), ["foundation", "uri", "wire-envelope", "lock-noalloc"])
        self.assertEqual(runner.select_suites(None, [], ["scene"]),
                         ["wire-scene", "scene-journal", "sdf-scene", "scene-reference"])
        self.assertEqual(runner.select_suites(None, ["native"], ["wire"]), ["wire-envelope", "usd-values"])
        self.assertEqual(runner.select_suites(None, ["manual"], []), ["kit-reference"])
        self.assertEqual(runner.select_suites("foundation,uri", [], []), ["foundation", "uri"])
        # Planned suites are selected only by name.
        self.assertFalse(runner.PLANNED & set(runner.select_suites(None, [], ["runtime", "scene", "plugin", "kit"])))

    def test_a_tag_selection_runs_every_selected_suite(self):
        status, _, manifest = self.run_main(["--tag", "pr", "--seed", "42"], foundation_process())
        self.assertEqual(status, 2)  # The native inputs are missing.
        self.assertEqual([r["suite"] for r in manifest["results"]], ["foundation", "uri", "wire-envelope", "lock-noalloc"])
        self.assertEqual(manifest["results"][0]["outcome"], "passed")

    def test_invalid_selections_are_rejected_before_running(self):
        for arguments in (["--suite", "foundation", "--tag", "pr"], ["--tag", "no-such-tag"],
                          ["--package", "no-such-package"], ["--tag", "manual", "--package", "wire"],
                          ["--suite", "wire-envelope", "--scenario", "anything"],
                          ["--suite", "foundation", "--shard", "2/2"], ["--suite", "foundation", "--scenario", ""],
                          ["--suite", "foundation", "--driver-mode", "actor-auth-timeout"],
                          ["--suite", "native-reference", "--driver-mode", "actor-auth-fail"],
                          ["--suite", "native-reference", "--driver-mode", "actor-auth-timeout", "--shard", "0/2"]):
            with self.subTest(arguments=arguments):
                self.rejected([*arguments, "--seed", "42"])

    def test_scenarios_and_shard_reach_the_suite_and_replay_restores_them(self):
        seen = []
        status, run, manifest = self.run_main(["--suite", "foundation", "--seed", "42", "--scenario",
                                               "test_a,test_b.Tests", "--shard", "1/4"], foundation_process(seen))
        self.assertEqual(status, 0)
        self.assertEqual(manifest["selection"], dict(scenarios=["test_a", "test_b.Tests"], shard="1/4", driver_mode=None))
        replay = json.loads((run / "replay.json").read_text())
        self.assertEqual((replay["scenarios"], replay["shard"], replay["driver_mode"]),
                         (["test_a", "test_b.Tests"], "1/4", None))
        status, _, replayed = self.run_main(["--replay", str(run / "replay.json")], foundation_process(seen))
        self.assertEqual((status, replayed["selection"]), (0, manifest["selection"]))
        for options in seen:
            self.assertEqual(json.loads(options["environment"]["LIVE_TEST_SCENARIOS"]), ["test_a", "test_b.Tests"])
            self.assertEqual(options["environment"]["LIVE_TEST_SHARD"], "1/4")
            self.assertTrue(options["scratch"])
            self.assertIsNone(options["ready"])  # Foundation starts no service.

    def test_no_selection_passes_no_selection_environment(self):
        seen = []
        self.run_main(["--suite", "foundation", "--seed", "42"], foundation_process(seen))
        self.assertNotIn("LIVE_TEST_SCENARIOS", seen[0]["environment"])
        self.assertNotIn("LIVE_TEST_SHARD", seen[0]["environment"])

    def test_results_carry_fixture_and_scenario_fingerprints(self):
        status, run, manifest = self.run_main(["--suite", "foundation,transport", "--seed", "42"], foundation_process())
        self.assertEqual(status, 2)
        foundation, transport = manifest["results"]
        self.assertEqual(foundation["fixture_fingerprint"], runner.fixture_fingerprint("foundation"))
        self.assertEqual(foundation["scenario_fingerprint"], runner.scenario_fingerprint("foundation", ["test_a.Tests.test_one"]))
        self.assertEqual((transport["outcome"], transport["reason"]), ("infrastructure_error", "suite not implemented"))
        self.assertEqual(transport["fixture_fingerprint"], runner.fixture_fingerprint("transport"))
        self.assertEqual(transport["scenario_fingerprint"], runner.scenario_fingerprint("transport"))
        replay = json.loads((run / "replay.json").read_text())
        self.assertEqual(replay["fixture_fingerprints"], {name: runner.fixture_fingerprint(name)
                                                          for name in ("foundation", "transport")})

    def test_planned_suite_fingerprints_follow_their_contracts_and_fixtures(self):
        scenarios = {name: runner.scenario_fingerprint(name) for name in runner.PLANNED}
        self.assertTrue(all(scenarios.values()))
        fixtures = runner.fixture_fingerprint("transport")
        catalog = copy.deepcopy(runner.CATALOG)
        catalog["suites"]["transport"]["contracts"] = catalog["suites"]["transport"]["contracts"][:-1]
        catalog["suites"]["transport"]["fixtures"] = catalog["suites"]["transport"]["fixtures"][:-1]
        with patch.object(runner, "CATALOG", catalog):
            self.assertNotEqual(runner.scenario_fingerprint("transport"), scenarios["transport"])
            self.assertNotEqual(runner.fixture_fingerprint("transport"), fixtures)
        # A contract's requirement text is part of the scenario fingerprint.
        contracts = json.loads((ROOT / "tests/contracts/catalog.json").read_text())
        contracts["contracts"][0]["requirement"] += " (changed)"
        with patch.object(Path, "read_text", lambda self, *a, **k: json.dumps(contracts)
                          if self.name == "catalog.json" else Path.read_bytes(self).decode()):
            self.assertNotEqual(runner.scenario_fingerprint("core"), scenarios["core"])

    def test_a_replay_with_changed_fixtures_is_refused(self):
        status, run, _ = self.run_main(["--suite", "foundation", "--seed", "42"], foundation_process())
        self.assertEqual(status, 0)
        with patch.object(runner, "fixture_fingerprint", return_value="changed-fixtures"):
            status, _, manifest = self.run_main(["--replay", str(run / "replay.json")], foundation_process())
        self.assertEqual(status, 2)
        self.assertEqual(manifest["results"][0]["reason"], "replay fixture fingerprint differs")

    def test_a_suite_over_its_resource_budget_fails(self):
        catalog = copy.deepcopy(runner.CATALOG)
        catalog["suites"]["foundation"]["budget"] = dict(artifact_bytes=4096, artifact_files=100)
        status, _, manifest = self.run_main(["--suite", "foundation", "--seed", "42"], foundation_process(
            extra=lambda directory: (directory / "flood.bin").write_bytes(b"x" * 8192)), catalog)
        self.assertEqual(status, 1)
        result = manifest["results"][0]
        self.assertEqual(result["outcome"], "failed")
        self.assertTrue(result["reason"].startswith("resource budget exceeded: artifact_bytes "), result["reason"])
        self.assertGreater(result["resource_usage"]["artifact_bytes"], 8192)
        catalog["suites"]["foundation"]["budget"] = dict(artifact_bytes=1 << 20, artifact_files=2)
        status, _, manifest = self.run_main(["--suite", "foundation", "--seed", "42"], foundation_process(
            extra=lambda directory: [(directory / f"{n}.txt").write_text("x") for n in range(3)]), catalog)
        self.assertEqual((status, manifest["results"][0]["outcome"]), (1, "failed"))
        self.assertIn("artifact_files", manifest["results"][0]["reason"])

    def test_the_suite_fixture_redaction_rewrites_its_log(self):
        catalog = copy.deepcopy(runner.CATALOG)
        catalog["suites"]["foundation"]["redact"] = ["bearer", "pem-private-key"]
        log = ("authorization: Bearer synthetic-writer.token/1+=\n"
               "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg\n-----END PRIVATE KEY-----\nOK\n")
        status, run, manifest = self.run_main(["--suite", "foundation", "--seed", "42"],
                                              foundation_process(log=log), catalog)
        self.assertEqual(status, 0)
        self.assertEqual((run / "0.log").read_text(), "authorization: Bearer [redacted]\n[redacted]\nOK\n")
        self.assertEqual(manifest["results"][0]["redactions"], 2)
        status, run, manifest = self.run_main(["--suite", "foundation", "--seed", "42"], foundation_process(log=log))
        self.assertEqual(((run / "0.log").read_text(), manifest["results"][0]["redactions"]), (log, 0))

    def test_redaction_patterns_keep_their_prefix_and_cover_every_match(self):
        patterns = [runner.CATALOG["redaction"][name] for name in ("bearer", "pem-private-key")]
        text, count = redact("BEARER a1 bearer b2 -----BEGIN RSA PRIVATE KEY-----\nx\n-----END RSA PRIVATE KEY-----", patterns)
        self.assertEqual((text, count), ("BEARER [redacted] bearer [redacted] [redacted]", 3))

    def test_an_interrupted_run_records_an_incomplete_manifest(self):
        def interrupt(command, **options):
            raise KeyboardInterrupt
        status, run, manifest = self.run_main(["--suite", "foundation,transport", "--seed", "42"], interrupt)
        self.assertEqual(status, 130)
        self.assertEqual((manifest["complete"], manifest["passed"]), (False, False))
        self.assertEqual([(r["suite"], r["outcome"]) for r in manifest["results"]], [("foundation", "interrupted")])
        self.assertTrue((run / "junit.xml").is_file())

    def test_a_platform_without_a_guard_refuses_every_suite(self):
        with patch.object(sys, "platform", "unsupported-os"):
            status, _, manifest = self.run_main(["--suite", "foundation", "--seed", "42"],
                                                lambda *a, **k: self.fail("launched a suite without containment"))
        self.assertEqual(status, 2)
        self.assertIn("unsupported-os is unsupported", manifest["results"][0]["reason"])

    def test_foundation_rejects_unexpected_platform_exclusions(self):
        def unexpected(directory):
            path = directory / 'foundation.json'
            summary = json.loads(path.read_text())
            summary['excluded'] = [dict(id='ordinary.product.test', reason='not supported')]
            path.write_text(json.dumps(summary))
        status, _, manifest = self.run_main(['--suite', 'foundation', '--seed', '42'],
                                            foundation_process(extra=unexpected))
        self.assertEqual(status, 2)
        self.assertEqual(manifest['results'][0]['outcome'], 'infrastructure_error')

    def test_list_names_planned_fingerprints(self):
        output = subprocess.run([sys.executable, "-B", str(ROOT / "tools/live/test.py"), "--list"],
                                capture_output=True, text=True, timeout=30, check=True).stdout
        for name in runner.PLANNED:
            self.assertIn(f"  {name}: package ", output)
            self.assertIn(f"scenarios {runner.scenario_fingerprint(name)[:12]}", output)

    def test_one_foundation_scenario_runs_alone(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = "test_selection.CaseSelectionTests.test_shard_text_is_strict"
            result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/live/test.py"), "--suite", "foundation",
                                     "--seed", "42", "--scenario", case, "--artifact-dir", temporary],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run = next(Path(temporary).iterdir())
            summary = json.loads((run / "foundation.json").read_text())
            self.assertEqual((summary["tests_run"], summary["test_ids"]), (1, [case]))
            self.assertGreater(summary["selection"]["available"], 100)
            missing = subprocess.run([sys.executable, "-B", str(ROOT / "tools/live/test.py"), "--suite", "foundation",
                                      "--seed", "42", "--scenario", "test_no_such_module", "--artifact-dir", temporary],
                                     capture_output=True, text=True, timeout=120)
            self.assertEqual(missing.returncode, 2, missing.stdout + missing.stderr)


if __name__ == "__main__":
    unittest.main()
