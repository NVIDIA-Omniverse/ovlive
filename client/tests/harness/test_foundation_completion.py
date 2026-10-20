# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Missing or skipped correctness work must never satisfy the CI foundation gate."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.live import foundation
from tools.live.foundation import MACOS_EXCLUSIONS, platform_suite, run


def named_case(identity):
    case = unittest.FunctionTestCase(lambda: None)
    case.id = lambda: identity
    return case


class FoundationCompletionTests(unittest.TestCase):
    def test_empty_and_duplicate_discovery_are_rejected(self):
        with self.assertRaises(ValueError):
            run(unittest.TestSuite(), io.StringIO())
        case = unittest.FunctionTestCase(lambda: None)
        with self.assertRaises(ValueError):
            run(unittest.TestSuite([case, case]), io.StringIO())

    def test_skipped_or_failing_work_cannot_pass(self):
        def skip():
            raise unittest.SkipTest('missing capability')
        def fail():
            raise AssertionError('injected failure')
        for fn in (skip, fail):
            with self.subTest(fn=fn.__name__):
                summary = run(unittest.TestSuite([unittest.FunctionTestCase(fn)]), io.StringIO())
                self.assertFalse(summary['passed'])
                self.assertEqual(summary['tests_run'], 1)

    def test_complete_passing_work_has_exact_count(self):
        summary = run(unittest.TestSuite([unittest.FunctionTestCase(lambda: None)]), io.StringIO())
        self.assertTrue(summary['passed'])
        self.assertEqual((summary['discovered'], summary['tests_run'], summary['skipped']), (1, 1, 0))

    def test_macos_excludes_exactly_detached_process_contracts_with_evidence(self):
        identities = [*MACOS_EXCLUSIONS, 'ordinary.contract']
        with patch('sys.platform', 'darwin'):
            selected, excluded = platform_suite(unittest.TestSuite(named_case(name) for name in identities))
            summary = run(selected, io.StringIO(), excluded=excluded)
        self.assertEqual(summary['test_ids'], ['ordinary.contract'])
        self.assertEqual(summary['excluded'], [dict(id=name, reason=MACOS_EXCLUSIONS[name]) for name in MACOS_EXCLUSIONS])
        self.assertEqual((summary['available'], summary['tests_run'], summary['skipped']), (5, 1, 0))
        self.assertEqual(summary['process_cleanup'], 'process-group')
        self.assertTrue(summary['passed'])
        for host in ('linux', 'win32'):
            with self.subTest(host=host), patch('sys.platform', host):
                selected, excluded = platform_suite(unittest.TestSuite(named_case(name) for name in identities))
                self.assertEqual(selected.countTestCases(), 5)
                self.assertEqual(excluded, [])

    def test_macos_exclusion_names_must_exist_in_discovery(self):
        with patch('sys.platform', 'darwin'), self.assertRaisesRegex(ValueError, 'Missing macOS exclusion cases'):
            platform_suite(unittest.TestSuite([named_case('ordinary.contract')]))

    def test_scenario_and_shard_evidence_retains_the_full_discovery_inventory(self):
        ordinary = [name + '.Tests.test_one' for name in sorted(foundation.REQUIRED_MODULES)]
        identities = [*MACOS_EXCLUSIONS, *ordinary]
        for host in ('darwin', 'linux', 'win32'):
            for scenarios, shard in (([ordinary[0]], None), ([], (0, 1)), ([], (1, 2))):
                with self.subTest(host=host, scenarios=scenarios, shard=shard), \
                        tempfile.TemporaryDirectory() as directory, \
                        patch('sys.platform', host), \
                        patch.object(unittest.defaultTestLoader, 'discover', return_value=unittest.TestSuite(
                            named_case(name) for name in identities)), \
                        patch.object(foundation, 'case_selection', return_value=(scenarios, shard)), \
                        patch.dict(os.environ, LIVE_TEST_ARTIFACT_DIR=directory), \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(foundation.main(), 0)
                    summary = json.loads((Path(directory) / 'foundation.json').read_text())
                self.assertEqual(summary['available'], len(identities))
                excluded = [dict(id=name, reason=reason) for name, reason in MACOS_EXCLUSIONS.items()] if host == 'darwin' else []
                self.assertEqual(summary['excluded'], excluded)
                self.assertEqual(summary['selection']['available'], len(identities) - len(excluded))
                self.assertEqual(summary['discovered'], summary['tests_run'])
