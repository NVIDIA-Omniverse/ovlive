#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed foundation discovery and explicit completion evidence."""
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live.test_support import case_selection, process_cleanup_scope, run_recording, select_cases, test_cases
REQUIRED_MODULES = {'test_foundation', 'test_abi_draft', 'test_live_build', 'test_runner'}
MACOS_EXCLUSIONS = {
    'test_foundation.SupervisorTests.test_successful_parent_reaps_detached_session':
        'requires cleanup of a descendant in a detached session',
    'test_foundation.SupervisorTests.test_outer_timeout_reaps_nested_supervised_process':
        'requires cleanup of a nested supervisor in another session',
    'test_supervision.RunnerDeathTests.test_an_interrupted_runner_leaves_no_process_or_directory':
        'requires cleanup of a detached grandchild after runner interruption',
    'test_supervision.RunnerDeathTests.test_a_crashed_runner_leaves_no_process_or_directory':
        'requires cleanup of a detached grandchild after runner death',
}


def platform_suite(suite):
    """Exclude only the named detached-process contracts on macOS; retain explicit exclusion evidence."""
    if sys.platform != 'darwin':
        return suite, []
    cases = list(test_cases(suite))
    missing = set(MACOS_EXCLUSIONS) - {case.id() for case in cases}
    if missing:
        raise ValueError('Missing macOS exclusion cases: ' + ', '.join(sorted(missing)))
    excluded = [dict(id=identity, reason=reason) for identity, reason in MACOS_EXCLUSIONS.items()]
    return unittest.TestSuite(case for case in cases if case.id() not in MACOS_EXCLUSIONS), excluded


def test_ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from test_ids(item)
        else:
            yield item.id()


def run(suite, stream=None, selection=None, excluded=(), available=None):
    identities = list(test_ids(suite))
    if not identities or len(identities) != len(set(identities)):
        raise ValueError('Foundation requires nonempty, unique discovered tests')
    if available is None:
        available = len(identities) + len(excluded)
    if type(available) is not int or available < len(identities) + len(excluded):
        raise ValueError('Foundation inventory cannot be smaller than selected and excluded tests')
    result, cases = run_recording(suite, stream=stream)
    return dict(schema_version=1, boundary='foundation', discovered=len(identities),
                available=available, excluded=list(excluded),
                process_cleanup=process_cleanup_scope(),
                test_ids=identities, cases=cases, tests_run=result.testsRun, skipped=len(result.skipped),
                expected_failures=len(result.expectedFailures), selection=selection,
                passed=(result.wasSuccessful() and not result.skipped and not result.expectedFailures
                        and result.testsRun == len(identities)))


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests/harness'))
    modules = {name.split('.')[0] for name in test_ids(suite)}
    if not REQUIRED_MODULES <= modules:
        raise ValueError('Missing required foundation modules: ' + ', '.join(sorted(REQUIRED_MODULES - modules)))
    available = suite.countTestCases()
    suite, excluded = platform_suite(suite)
    scenarios, shard = case_selection()
    selection = None
    if scenarios or shard:
        # The complete inventory is checked above; the selection runs a subset of it.
        selection = dict(scenarios=scenarios, shard=None if shard is None else f'{shard[0]}/{shard[1]}',
                         available=suite.countTestCases())
        try:
            suite = select_cases(suite, scenarios, shard)
        except ValueError as error:
            print(f'foundation selection invalid: {error}', file=sys.stderr)
            return 125  # An infrastructure error, never a test result.
    summary = run(suite, selection=selection, excluded=excluded, available=available)
    directory = Path(os.environ['LIVE_TEST_ARTIFACT_DIR'])
    (directory / 'foundation.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
