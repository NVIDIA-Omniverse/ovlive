# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence lines and exit status of the CI and test-run summarizer."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.live import summarize


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_ci_output(self, passed=True, campaign_passed=True):
        output = self.root / 'ci'
        write_json(output / 'summary.json', dict(
            schema_version=1, passed=passed, complete=True, profile='nightly', seed=42,
            source_fingerprint='f' * 8, checkout=dict(state='clean', changed_files=0),
            release_archive=dict(name='ovlive.tar.gz', sha256='a' * 8, git_head='c' * 8, dirty=False)))
        write_json(output / 'plan.json', dict(build_plan=dict(usd_profile='kit-25.11')))
        run = output / 'tests' / 'run-one'
        write_json(run / 'manifest.json', dict(
            schema_version=1, passed=passed, complete=True, git_head='c' * 8, source_fingerprint='f' * 8,
            seed=dict(schema_version=1, master_seed=42, writer_seeds=[5]),
            results=[dict(suite='foundation', outcome='passed', tests_run=3),
                     dict(suite='lock-noalloc', outcome='passed'),
                     dict(suite='uri', outcome='passed' if passed else 'failed')]))
        (run / '2.log').write_text('...\nRan 1 test in 0.1s\n...\nRan 4 tests in 0.2s\n', encoding='utf-8')
        write_json(run / 'writer-fuzz' / 'summary.json', dict(
            passed=campaign_passed, plan=dict(cases=[1, 2], seeds=[0, 42]), results=[1, 2]))
        (output / 'support-identity.json').write_bytes(b'{}')
        return output

    def run_main(self, directory):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = summarize.main([str(directory)])
        return status, stdout.getvalue().splitlines(), stderr.getvalue()

    def test_passing_ci_output_prints_every_evidence_line_and_exits_zero(self):
        status, lines, _ = self.run_main(self.make_ci_output())
        self.assertEqual(status, 0)
        self.assertEqual(lines, [
            'ci passed True complete True reason -',
            'ci profile nightly usd_profile kit-25.11 seed 42 fingerprint ffffffff',
            'checkout clean changed_files 0',
            'release archive ovlive.tar.gz sha256 aaaaaaaa git_head cccccccc dirty False',
            'run run-one passed True complete True git_head cccccccc fingerprint ffffffff',
            '  seed 42 writer_seeds 5',
            '  suite foundation passed tests 3',
            '  suite lock-noalloc passed tests -',
            '  suite uri passed tests 4',
            '  writer-fuzz passed True cases 2 results 2 seeds 0 42',
            'support identity sha256 ' + hashlib.sha256(b'{}').hexdigest(),
        ])

    def test_failed_ci_output_exits_one(self):
        status, lines, _ = self.run_main(self.make_ci_output(passed=False))
        self.assertEqual(status, 1)
        self.assertIn('  suite uri failed tests 4', lines)

    def test_failed_writer_campaign_exits_one(self):
        status, _, _ = self.run_main(self.make_ci_output(campaign_passed=False))
        self.assertEqual(status, 1)

    def test_test_artifact_directory_prints_its_runs(self):
        output = self.make_ci_output()
        status, lines, _ = self.run_main(output / 'tests')
        self.assertEqual(status, 0)
        self.assertEqual(lines[0], 'run run-one passed True complete True git_head cccccccc fingerprint ffffffff')
        self.assertFalse(any(line.startswith(('ci ', 'support ')) for line in lines))

    def test_usd_profile_falls_back_to_the_support_identity(self):
        output = self.make_ci_output()
        (output / 'plan.json').unlink()
        (output / 'support-identity.json').write_text(json.dumps(dict(usd_profile='stock-25.11')), encoding='utf-8')
        status, lines, _ = self.run_main(output)
        self.assertEqual(status, 0)
        self.assertEqual(lines[1], 'ci profile nightly usd_profile stock-25.11 seed 42 fingerprint ffffffff')

    def test_run_without_a_manifest_is_incomplete_and_exits_one(self):
        output = self.make_ci_output()
        (output / 'tests' / 'run-early').mkdir()
        status, lines, stderr = self.run_main(output / 'tests')
        self.assertEqual(status, 1)
        self.assertEqual(lines[0], 'run run-early manifest missing or unreadable')
        self.assertEqual(stderr, '')

    def test_incomplete_run_exits_one(self):
        run = self.root / 'runs' / 'run-one'
        write_json(run / 'manifest.json', dict(schema_version=1, complete=False, git_head='c' * 8,
                                               source_fingerprint='f' * 8, results=[]))
        status, lines, _ = self.run_main(self.root / 'runs')
        self.assertEqual(status, 1)
        self.assertEqual(lines, ['run run-one passed None complete False git_head cccccccc fingerprint ffffffff'])

    def test_runs_are_listed_in_name_order(self):
        output = self.make_ci_output()
        (output / 'tests' / 'run-one').rename(output / 'tests' / 'run-b')
        self.make_ci_output()
        status, lines, _ = self.run_main(output / 'tests')
        self.assertEqual(status, 0)
        self.assertEqual([line.split()[1] for line in lines if line.startswith('run ')], ['run-b', 'run-one'])

    def test_directory_without_results_exits_three(self):
        status, lines, stderr = self.run_main(self.root)
        self.assertEqual(status, 3)
        self.assertEqual(lines, [])
        self.assertIn('no CI summary or run-* directory', stderr)


if __name__ == '__main__':
    unittest.main()
