# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone replay of recorded graph, rotation, period and staged-samples reduced writer plans: the recipe a reduction
records, the checks before any trial and the runner's exit statuses; the plugin-reference and scene-reference suites
and the package campaign's self-test case replay real self-test reductions."""
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from tools.live import reference
from tools.live import test as runner
from tools.live.test_support import ProcessResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
import period_plans
import plan_replay
import rotation_history
import sample_history
import staged_sample_history
import structural_history
from writer_reduction import digest, period_domain, reduce_history, round_domain, staged_domain

CANDIDATE = {'dependencies': [{'role': role, 'path': str(Path(role).resolve())}
                              for role in ('plug_info', 'diagnostics_sample', 'participant_sample',
                                           'participant_tests')]}
# The identities the mocked runner and lane report: no sanitizer, and `{}` for the reference identity.
IDENTITIES = dict(source_fingerprint='source', candidate=CANDIDATE, reference_identity={}, sanitizer_runtime={})
# The stage identity a staged-samples reduction records and the mocked runner and lane report.
STAGE = dict(path=str(Path('stage').resolve()), manifest_sha256='0' * 64,
             interpreter=dict(path='python3.12', sha256='1' * 64), samples={'bin/live_participant_sample': '2' * 64})


def plans():
    """One seeded plan, its named test-only defect, its domain and the edit the defect needs, per class."""
    return {
        'structural': (dict(structural_history.generate(42, clients=2), test_service_fault='corrupt-reorder-w1:2'),
                       round_domain(structural_history.validate, structural_history.origin), 'w1:2'),
        'samples': (dict(sample_history.generate(42, clients=2, rounds=12), test_service_fault='corrupt-sample-w0:6'),
                    round_domain(sample_history.validate, sample_history.origin), 'w0:6'),
        'uncertain-before': (dict(rotation_history.generate_uncertain(42), test_service_fault='apply-cut-w0:0'),
                             round_domain(rotation_history.validate, rotation_history.origin), 'w0:0'),
        'stream-age': (dict(period_plans.generate(42, 4, 8, period_plans.STREAM_AGE), test_service_fault='corrupt-w0:4'),
                       period_domain(), 'w0:4'),
        'staged-samples': (dict(staged_sample_history.generate(42, 2, 4),
                                test_service_fault=staged_sample_history.CORRUPT), staged_domain(), 'w0:2'),
    }


TARGET = dict(reason='synthetic', period=5)


def record(root, kind, scale=None):
    """Reduce the class's plan with a model evaluator that fails while the defect's edit is retained, recording the
    replay context; returns the reduction directory."""
    plan, domain, edit = plans()[kind]

    async def evaluate(candidate, directory):
        retained = edit in [op['edit'] for writer in candidate['writers'] for op in writer]
        return (dict(outcome='mismatch', signature=TARGET, cleanup_complete=True) if retained else
                dict(outcome='passed', cleanup_complete=True))
    lane = plan_replay.KINDS[kind][0]
    context = dict(kind=kind, lane=lane, scale=scale, **IDENTITIES,
                   **({'staged_package': STAGE} if kind in plan_replay.STAGED_KINDS else {}))
    summary = asyncio.run(reduce_history(plan, TARGET, root / f'{kind}-reduction', evaluate, domain=domain,
                                         context=context))
    assert summary['status'] == 'complete', summary
    return root / f'{kind}-reduction'


def rewrite(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


class RecipeTests(unittest.TestCase):
    def test_a_reduction_records_its_replay_context_and_each_class_loads_both_plans(self):
        with tempfile.TemporaryDirectory() as temporary:
            for kind in plans():
                with self.subTest(kind=kind):
                    root = record(Path(temporary), kind)
                    summary = json.loads((root / 'reduction.json').read_text())
                    self.assertEqual(summary['context']['kind'], kind)
                    for name, key in plan_replay.PLANS.items():
                        recipe = plan_replay.load(root / name)
                        plan = json.loads((root / name).read_text())
                        self.assertEqual((recipe['kind'], recipe['lane'], recipe['name']),
                                         (kind, plan_replay.KINDS[kind][0], name))
                        self.assertEqual((recipe['plan'], recipe['plan_sha256']), (plan, summary[key]))
                        self.assertEqual(recipe['target'], TARGET)
                        self.assertIn('test_service_fault', recipe['plan'])
                    best = plan_replay.load(root / 'best.plan.json')
                    self.assertLess(sum(map(len, best['plan']['writers'])),
                                    sum(map(len, plan_replay.load(root / 'original.plan.json')['plan']['writers'])))

    def test_an_altered_plan_or_recorded_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            for kind in plans():
                root = record(Path(temporary), kind)
                for name, key in plan_replay.PLANS.items():
                    plan_bytes, summary_bytes = (root / name).read_bytes(), (root / 'reduction.json').read_bytes()
                    for alteration in ('plan', 'hash'):
                        with self.subTest(kind=kind, name=name, alteration=alteration):
                            # Removing the defect leaves a valid seeded plan, which only the recorded hash refuses.
                            if alteration == 'plan':
                                rewrite(root / name, lambda plan: plan.pop('test_service_fault'))
                            else:
                                rewrite(root / 'reduction.json', lambda summary: summary.update({key: '0' * 64}))
                            with self.assertRaisesRegex(ValueError, 'differs from the hash'):
                                plan_replay.load(root / name)
                            (root / name).write_bytes(plan_bytes)
                            (root / 'reduction.json').write_bytes(summary_bytes)
                            self.assertEqual(plan_replay.load(root / name)['name'], name)

    def test_only_a_recorded_reduction_of_a_replayable_class_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = record(Path(temporary), 'uncertain-before')
            with self.assertRaisesRegex(ValueError, 'best.plan.json or original.plan.json'):
                plan_replay.load(root / 'trial-0001/input.plan.json')
            summary = (root / 'reduction.json').read_bytes()
            for change in (lambda s: s.pop('context'), lambda s: s['context'].update(kind='root-fields'),
                           lambda s: s['context'].update(lane='scene-reference'),
                           lambda s: s['context'].pop('sanitizer_runtime'), lambda s: s.update(schema_version=1),
                           lambda s: s.update(target=None), lambda s: s['context'].update(scale={'warmup': 1})):
                with self.subTest(change=change):
                    rewrite(root / 'reduction.json', change)
                    with self.assertRaises(ValueError):
                        plan_replay.load(root / 'best.plan.json')
                    (root / 'reduction.json').write_bytes(summary)
            # A uncertain-write plan recorded as a queued-rotation case is not that case's plan.
            rewrite(root / 'reduction.json', lambda s: s['context'].update(kind='queued-rotations'))
            with self.assertRaisesRegex(ValueError, 'not a queued-rotations plan'):
                plan_replay.load(root / 'best.plan.json')

    def test_a_stream_age_recipe_keeps_its_case_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            scale = dict(warmup=10, thresholds=True, slope=False)
            root = record(Path(temporary), 'stream-age', scale)
            self.assertEqual(plan_replay.load(root / 'best.plan.json')['scale'], scale)

    def test_a_malformed_stream_age_scale_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = record(Path(temporary), 'stream-age', dict(warmup=1, thresholds=False, slope=False))
            summary = (root / 'reduction.json').read_bytes()
            for scale in ({}, dict(warmup=1, thresholds=False), dict(warmup='1', thresholds=False, slope=False),
                          dict(warmup=0, thresholds=False, slope=False), dict(warmup=1, thresholds=1, slope=False),
                          dict(warmup=1, thresholds=False, slope=None),
                          dict(warmup=1, thresholds=False, slope=False, extra=True), [1, False, False]):
                with self.subTest(scale=scale):
                    rewrite(root / 'reduction.json', lambda s: s['context'].update(scale=scale))
                    with self.assertRaisesRegex(ValueError, 'invalid reduction case scale'):
                        plan_replay.load(root / 'best.plan.json')
                    (root / 'reduction.json').write_bytes(summary)

    def test_a_lane_outside_the_runner_records_absent_identities_and_its_record_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver = Path(temporary) / 'driver'
            Path(str(driver) + '.manifest.json').write_text(json.dumps(CANDIDATE))
            complete = {'LIVE_TEST_SCENE_DRIVER': str(driver), 'LIVE_TEST_SOURCE_FINGERPRINT': 'source',
                        'LIVE_TEST_REFERENCE_IDENTITY': '{}'}
            self.assertEqual(plan_replay.context('structural', environment=complete),
                             dict(kind='structural', lane='scene-reference', scale=None, **IDENTITIES))
            for missing, key in (('LIVE_TEST_SOURCE_FINGERPRINT', 'source_fingerprint'),
                                 ('LIVE_TEST_REFERENCE_IDENTITY', 'reference_identity'),
                                 ('LIVE_TEST_SCENE_DRIVER', 'candidate')):
                with self.subTest(missing=missing):
                    environment = {name: value for name, value in complete.items() if name != missing}
                    context = plan_replay.context('structural', environment=environment)
                    self.assertIsNone(context[key])
                    self.assertTrue(all(context[other] is not None for other in plan_replay.IDENTITIES
                                        if other != key))
                    # The reduction still runs and records the context; only its replay is refused.
                    root = record(Path(temporary) / missing, 'structural')
                    rewrite(root / 'reduction.json', lambda s: s.update(context=context))
                    with self.assertRaisesRegex(ValueError, 'recorded no identity'):
                        plan_replay.load(root / 'best.plan.json')
            Path(str(driver) + '.manifest.json').write_text('not json')
            self.assertIsNone(plan_replay.context('structural', environment=complete)['candidate'])

    def test_without_the_defect_only_removes_a_named_defect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = record(Path(temporary), 'samples')
            recipe = plan_replay.load(root / 'best.plan.json', without_defect=True)
            self.assertNotIn('test_service_fault', recipe['plan'])
            self.assertTrue(recipe['without_test_defect'])
            self.assertEqual(recipe['plan_sha256'], digest(recipe['plan']))
            rewrite(root / 'best.plan.json', lambda plan: plan.pop('test_service_fault'))
            rewrite(root / 'reduction.json', lambda summary: summary.update(
                best_sha256=digest(json.loads((root / 'best.plan.json').read_text()))))
            with self.assertRaisesRegex(ValueError, 'no named test-only defect'):
                plan_replay.load(root / 'best.plan.json', without_defect=True)

    def test_a_retained_structural_v1_plan_is_refused(self):
        """A structural plan recorded before the seeded generator, with its recorded hash, is refused, not misread."""
        with tempfile.TemporaryDirectory() as temporary:
            root = record(Path(temporary), 'structural')
            rewrite(root / 'best.plan.json', lambda plan: plan.update(scenario='private-attachment.structural-v1',
                                                                      schema_version=1))
            rewrite(root / 'reduction.json', lambda summary: summary.update(
                best_sha256=digest(json.loads((root / 'best.plan.json').read_text()))))
            with self.assertRaisesRegex(ValueError, 'not a structural plan'):
                plan_replay.load(root / 'best.plan.json')

    def test_every_recorded_identity_must_equal_the_current_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            recipe = plan_replay.load(record(Path(temporary), 'structural') / 'best.plan.json')
            current = dict(source='source', candidate=CANDIDATE, reference={}, sanitizer={})
            self.assertIs(plan_replay.verify(recipe, **current), recipe)
            for key in current:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identity differs'):
                    plan_replay.verify(recipe, **dict(current, **{key: {'stale': True}}))

    def test_a_staged_samples_recipe_requires_and_verifies_its_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = record(Path(temporary), 'staged-samples')
            recipe = plan_replay.load(root / 'best.plan.json')
            self.assertEqual(recipe['staged_package'], STAGE)
            self.assertEqual(plan_replay.identities('staged-samples'), (*plan_replay.IDENTITIES, 'staged_package'))
            current = dict(source='source', candidate=CANDIDATE, reference={}, sanitizer={})
            self.assertIs(plan_replay.verify(recipe, **current, staged=STAGE), recipe)
            # A changed stage path, interpreter or sample executable, or no stage, is refused.
            for staged in (dict(STAGE, path='/elsewhere'), dict(STAGE, interpreter=dict(STAGE['interpreter'],
                                                                                          sha256='3' * 64)),
                           dict(STAGE, samples={'bin/live_participant_sample': '4' * 64}), None):
                with self.subTest(staged=staged), self.assertRaisesRegex(ValueError, 'stage'):
                    plan_replay.verify(recipe, **current, staged=staged)
            # Any other class names no stage.
            other = plan_replay.load(record(Path(temporary), 'stream-age') / 'best.plan.json')
            self.assertNotIn('staged_package', other)
            with self.assertRaisesRegex(ValueError, 'runs no staged package'):
                plan_replay.verify(other, **current, staged=STAGE)
            summary = (root / 'reduction.json').read_bytes()
            for change, message in ((lambda s: s['context'].pop('staged_package'), 'no replay context'),
                                    (lambda s: s['context'].update(staged_package=None), 'recorded no identity')):
                with self.subTest(message=message):
                    rewrite(root / 'reduction.json', change)
                    with self.assertRaisesRegex(ValueError, message):
                        plan_replay.load(root / 'best.plan.json')
                    (root / 'reduction.json').write_bytes(summary)

    def test_a_staged_samples_context_records_the_lanes_stage(self):
        environment = {'LIVE_TEST_SOURCE_FINGERPRINT': 'source', 'LIVE_TEST_REFERENCE_IDENTITY': '{}'}
        with patch.object(plan_replay, 'staged_identity', return_value=STAGE) as identity:
            self.assertEqual(plan_replay.context('staged-samples', environment=dict(
                environment, LIVE_TEST_STAGED_PACKAGE='stage'))['staged_package'], STAGE)
            identity.assert_called_once_with('stage')
            self.assertIsNone(plan_replay.context('staged-samples', environment=environment)['staged_package'])
            self.assertNotIn('staged_package', plan_replay.context('stream-age', environment=dict(
                environment, LIVE_TEST_STAGED_PACKAGE='stage')))
        with patch.object(plan_replay, 'staged_identity', side_effect=ValueError('Package member hash mismatch')):
            self.assertIsNone(plan_replay.context('staged-samples', environment=dict(
                environment, LIVE_TEST_STAGED_PACKAGE='stage'))['staged_package'])

    def test_only_the_recorded_signature_with_complete_cleanup_reproduces(self):
        result = dict(outcome='mismatch', signature=dict(TARGET), cleanup_complete=True)
        self.assertTrue(plan_replay.reproduced(result, TARGET))
        for other in (dict(result, signature=dict(TARGET, period=6)), dict(result, cleanup_complete=False),
                      dict(outcome='passed', cleanup_complete=True), dict(outcome='unresolved', mismatch=TARGET),
                      None):
            with self.subTest(other=other):
                self.assertFalse(plan_replay.reproduced(other, TARGET))


class RunnerTests(unittest.TestCase):
    """The common runner's reduced-plan mode, with the identities and the lane mocked."""

    def run_runner(self, root, plan, *extra, lane=None):
        """Run test.py on `plan`; `lane(command)` writes the lane's record and returns its ProcessResult, or None when
        no lane may launch. Returns the exit status and the launched commands."""
        commands = []

        def execute(command, **options):
            commands.append(command)
            return lane(command, Path(options['environment']['LIVE_TEST_ARTIFACT_DIR']))
        argv = ['test.py', '--reduced-plan', str(plan), '--artifact-dir', str(root / 'runs'),
                '--reference-python', sys.executable, '--reference-openssl', sys.executable,
                '--plugin-driver', 'plugin-driver', '--scene-transport-driver', 'scene-driver', *extra]
        with patch.object(sys, 'argv', argv), \
                patch.object(runner, 'suite_platform_error', return_value=None), \
                patch.object(runner, 'fingerprint', return_value='source'), \
                patch.object(runner, 'native_identity', return_value=CANDIDATE), \
                patch.object(plan_replay, 'staged_identity', side_effect=lambda path: dict(
                    STAGE, path=str(Path(path).resolve()))), \
                patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(stdout='{}')), \
                patch.object(runner, 'run_process', side_effect=execute), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            try:
                code = runner.main()
            except SystemExit as error:
                code = error.code
        self.errors = errors.getvalue()
        return code, commands

    @staticmethod
    def lane(trial, returncode=None):
        def run(command, directory):
            recipe = plan_replay.load(Path(command[command.index('--reduced-plan') + 1]),
                                      without_defect='--without-test-defect' in command)
            evidence = plan_replay.evidence(recipe, trial)
            (directory / 'reduced-plan-result.json').write_text(json.dumps(evidence))
            code = returncode if returncode is not None else 1 if evidence['reproduced'] else 2
            return ProcessResult('failed', code, 0, 0, False, True)
        return run

    def test_a_reproduced_signature_exits_1_and_any_other_valid_trial_exits_2(self):
        mismatch = dict(outcome='mismatch', signature=TARGET, cleanup_complete=True)
        for kind in ('structural', 'uncertain-before', 'stream-age'):
            for trial, extra, expected in ((mismatch, (), 1),
                                           (dict(outcome='passed', cleanup_complete=True), ('--without-test-defect',), 2),
                                           (dict(mismatch, signature=dict(TARGET, period=6)), (), 2),
                                           (dict(outcome='unresolved', cleanup_complete=False), (), 2)):
                with self.subTest(kind=kind, trial=trial), tempfile.TemporaryDirectory() as temporary:
                    root = record(Path(temporary), kind)
                    code, commands = self.run_runner(Path(temporary), root / 'best.plan.json', *extra,
                                                     lane=self.lane(trial))
                    self.assertEqual(code, expected, self.errors)
                    self.assertEqual(len(commands), 1)
                    lane = plan_replay.KINDS[kind][0]
                    self.assertIn('--plugin-driver' if lane == 'plugin-reference' else '--scene-transport-driver',
                                  commands[0])
                    self.assertEqual('--without-test-defect' in commands[0], bool(extra))
                    run = next((Path(temporary) / 'runs').iterdir())
                    copy = Path(commands[0][commands[0].index('--reduced-plan') + 1])
                    self.assertEqual(copy, run / 'reduced-plan/best.plan.json')
                    manifest = json.loads((run / 'manifest.json').read_text())
                    self.assertEqual([entry['suite'] for entry in manifest['results']], [lane])
                    self.assertEqual(manifest['results'][0]['reduced_plan']['reproduced'], expected == 1)
                    replay = json.loads((run / 'replay.json').read_text())
                    self.assertEqual(replay['reduced_plan'], dict(path=str(copy), without_test_defect=bool(extra)))
                    self.assertEqual(replay['master_seed'], 42)

    def test_a_staged_samples_replay_names_its_stage_to_the_lane_and_refuses_another(self):
        mismatch = dict(outcome='mismatch', signature=TARGET, cleanup_complete=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = record(Path(temporary), 'staged-samples')
            code, commands = self.run_runner(Path(temporary), root / 'best.plan.json', '--staged-package', 'stage',
                                             lane=self.lane(mismatch))
            self.assertEqual((code, len(commands)), (1, 1), self.errors)
            self.assertEqual(commands[0][commands[0].index('--staged-package') + 1], STAGE['path'])
            self.assertIn('--plugin-driver', commands[0])
            run = next((Path(temporary) / 'runs').iterdir())
            replay = json.loads((run / 'replay.json').read_text())
            self.assertEqual((replay['staged_package'], replay['staged_package_identity']), (STAGE['path'], STAGE))
            # Another stage is refused before the lane launches, and a replay without one is a usage error.
            code, commands = self.run_runner(Path(temporary), root / 'best.plan.json', '--staged-package', 'other',
                                             lane=None)
            self.assertEqual((code, commands), (3, []))
            code, commands = self.run_runner(Path(temporary), root / 'best.plan.json', lane=None)
            self.assertEqual((code, commands), (3, []))
            self.assertIn('--staged-package is required by, and only by', self.errors)
            # A class without a stage refuses one.
            other = record(Path(temporary), 'stream-age')
            code, commands = self.run_runner(Path(temporary), other / 'best.plan.json', '--staged-package', 'stage',
                                             lane=None)
            self.assertEqual((code, commands), (3, []))

    def test_a_lane_status_that_disagrees_with_its_record_exits_3(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = record(Path(temporary), 'uncertain-before')
            trial = dict(outcome='mismatch', signature=TARGET, cleanup_complete=True)
            code, commands = self.run_runner(Path(temporary), root / 'best.plan.json', lane=self.lane(trial, 2))
            self.assertEqual((code, len(commands)), (3, 1))

    def test_an_altered_plan_or_hash_is_refused_with_3_before_a_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = record(Path(temporary), 'stream-age')
            plan, summary = (root / 'best.plan.json').read_bytes(), (root / 'reduction.json').read_bytes()
            for change in (lambda: rewrite(root / 'best.plan.json', lambda p: p.pop('test_service_fault')),
                           lambda: rewrite(root / 'reduction.json', lambda s: s.update(best_sha256='0' * 64))):
                change()
                code, commands = self.run_runner(Path(temporary), root / 'best.plan.json', lane=None)
                self.assertEqual((code, commands), (3, []))
                self.assertIn('differs from the hash', self.errors)
                self.assertFalse((Path(temporary) / 'runs').exists())
                (root / 'best.plan.json').write_bytes(plan)
                (root / 'reduction.json').write_bytes(summary)

    def test_a_malformed_scale_or_absent_identity_is_refused_with_3_before_a_run_directory(self):
        for change in (lambda s: s['context'].update(scale={}), lambda s: s['context'].update(source_fingerprint=None)):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = record(Path(temporary), 'stream-age', dict(warmup=1, thresholds=False, slope=False))
                rewrite(root / 'reduction.json', change)
                code, commands = self.run_runner(Path(temporary), root / 'original.plan.json', lane=None)
                self.assertEqual((code, commands), (3, []))
                self.assertFalse((Path(temporary) / 'runs').exists())

    def test_a_changed_identity_is_refused_with_3_before_the_lane_launches(self):
        for key in plan_replay.IDENTITIES:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = record(Path(temporary), 'samples')
                rewrite(root / 'reduction.json', lambda s: s['context'].update({key: {'stale': True}}))
                code, commands = self.run_runner(Path(temporary), root / 'original.plan.json', lane=None)
                self.assertEqual((code, commands), (3, []))
                run = next((Path(temporary) / 'runs').iterdir())
                manifest = json.loads((run / 'manifest.json').read_text())
                self.assertIn('identity differs', manifest['results'][0]['reason'])

    def test_usage_errors_of_a_reduced_plan_replay_exit_3(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = record(Path(temporary), 'structural')
            for extra in (('--suite', 'plugin-reference'), ('--scenario', 'test_x'), ('--writer-plan', 'x.json'),
                          ('--unknown-option',)):
                with self.subTest(extra=extra):
                    code, commands = self.run_runner(Path(temporary), root / 'best.plan.json', *extra, lane=None)
                    self.assertEqual((code, commands), (3, []))
            code, _ = self.run_runner(Path(temporary), root / 'trial-0001/input.plan.json', lane=None)
            self.assertEqual(code, 3)
            # An abbreviation argparse resolves to --reduced-plan alone is the same mode.
            for option in ('--reduced', '--red', '--reduced-plan'):
                for arguments in ([option, str(root / 'best.plan.json'), '--bogus'],
                                  [f'{option}={root / "best.plan.json"}', '--bogus']):
                    with self.subTest(arguments=arguments), patch.object(sys, 'argv', ['test.py', *arguments]), \
                            contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as rejected:
                        runner.main()
                    self.assertEqual(rejected.exception.code, 3)
            # A prefix that also names another option is not the reduced-plan mode.
            with patch.object(sys, 'argv', ['test.py', '--re', 'x', '--bogus']), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as rejected:
                runner.main()
            self.assertEqual(rejected.exception.code, 2)
            # Without a reduced plan, a usage error keeps the runner's own status.
            with patch.object(sys, 'argv', ['test.py', '--without-test-defect']), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as rejected:
                runner.main()
            self.assertEqual(rejected.exception.code, 2)


class LaneTests(unittest.TestCase):
    """reference.py, the lane, checks the recipe before its service starts."""

    def test_the_lane_refuses_a_stale_recipe_before_sockets(self):
        for fault in ('source', 'candidate', 'reference', 'lane', 'hash'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                root = record(Path(temporary), 'uncertain-before')
                if fault == 'hash':
                    rewrite(root / 'reduction.json', lambda s: s.update(best_sha256='0' * 64))
                driver = ['--scene-transport-driver', 'driver'] if fault == 'lane' else [
                    '--plugin-driver', 'driver', '--plugin-plug-info', 'plug_info', '--diagnostics-sample',
                    'diagnostics_sample']
                arguments = ['reference.py', '--openssl', 'openssl', *driver, '--reduced-plan',
                             str(root / 'best.plan.json')]
                probe = MagicMock()
                with patch.object(sys, 'argv', arguments), \
                        patch.dict(os.environ, {'LIVE_TEST_ARTIFACT_DIR': temporary}, clear=True), \
                        patch.object(reference, 'identity', return_value={'stale': True} if fault == 'reference'
                                     else {}), \
                        patch.object(runner, 'fingerprint', return_value='stale' if fault == 'source' else 'source'), \
                        patch.object(runner, 'native_identity', return_value={'stale': True} if fault == 'candidate'
                                     else CANDIDATE), \
                        patch.object(reference, 'socket', SimpleNamespace(socket=probe)), \
                        patch.object(reference.subprocess, 'run') as compiler, \
                        patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stderr(io.StringIO()) as errors:
                    self.assertEqual(reference.main(), 125)
                    self.assertIn('Reference-service capability unavailable', errors.getvalue())
                    probe.assert_not_called()
                    compiler.assert_not_called()

    def test_the_lane_refuses_a_changed_or_missing_stage_before_sockets(self):
        for stage in ('other', None):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                root = record(Path(temporary), 'staged-samples')
                arguments = ['reference.py', '--openssl', 'openssl', '--plugin-driver', 'driver', '--plugin-plug-info',
                             'plug_info', '--diagnostics-sample', 'diagnostics_sample', '--reduced-plan',
                             str(root / 'best.plan.json'), *(['--staged-package', stage] if stage else [])]
                probe = MagicMock()
                with patch.object(sys, 'argv', arguments), \
                        patch.dict(os.environ, {'LIVE_TEST_ARTIFACT_DIR': temporary}, clear=True), \
                        patch.object(reference, 'identity', return_value={}), \
                        patch.object(runner, 'fingerprint', return_value='source'), \
                        patch.object(runner, 'native_identity', return_value=CANDIDATE), \
                        patch.object(plan_replay, 'staged_identity', side_effect=lambda path: dict(
                            STAGE, path=str(Path(path).resolve()))), \
                        patch.object(reference, 'socket', SimpleNamespace(socket=probe)), \
                        patch.object(reference.subprocess, 'run') as compiler, \
                        patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stderr(io.StringIO()) as errors:
                    self.assertEqual(reference.main(), 125)
                    self.assertIn('stage', errors.getvalue())
                    probe.assert_not_called()
                    compiler.assert_not_called()

    def test_reduced_plan_options_need_one_plugin_or_scene_driver(self):
        for arguments in (['--reduced-plan', 'best.plan.json'], ['--without-test-defect', '--plugin-driver', 'd'],
                          ['--reduced-plan', 'best.plan.json', '--plugin-driver', 'd', '--writer-plan', 'w']):
            with self.subTest(arguments=arguments), \
                    patch.object(sys, 'argv', ['reference.py', '--openssl', 'openssl', *arguments]), \
                    patch.object(reference, 'identity') as identity, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as rejected:
                    reference.main()
                self.assertEqual(rejected.exception.code, 2)
                identity.assert_not_called()


if __name__ == '__main__':
    unittest.main()
