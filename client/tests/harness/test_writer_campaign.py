# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Campaign coverage, refusal and supervision contracts with immediate outcomes."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live.test_support import ProcessResult, write_json
from tools.live.writer_campaign import (LONG_ROTATION_TIMEOUT_MS, LONG_ROTATIONS, OBSERVER_PEAK, PROFILES, SCENARIOS,
                                        SOAK_PAUSE_MS, SOAK_ROTATIONS, SOAK_TIMEOUT_MS, STREAM_AGE_PERIODS,
                                        STREAM_AGE_TIMEOUTS_MS, admit_client, case_passed, generate, observe_workload,
                                        replay_context, validate_case, verify_context, verify_summary)
from tools.live.writer_fuzz import artifact_bytes, execute


def completed(case):
    return dict(schema_version=1, case=case, tests_run=1, passed=True,
                peak_clients=case['clients'] + OBSERVER_PEAK[case['scenario']],
                workload=dict(seed=case['seed'], clients=case['clients'], steps=[case['steps']] * case['clients']),
                skipped=0, errors=0, failures=0, unexpected_successes=0, expected_failures=0)


def runner(command, *, output, environment, **kwargs):
    directory = output.parent
    recipe = json.loads(Path(command[-1]).read_text())
    case = recipe['case']
    assert environment['LIVE_TEST_SEED'] == str(case['seed'])
    assert environment['LIVE_TEST_ARTIFACT_DIR'] == str(directory)
    pending = json.loads((directory.parent / 'summary.json').read_text())
    assert pending['results'][-1]['process']['outcome'] == 'pending'
    write_json(directory / 'writer-case-result.json', completed(case))
    output.write_text('completed\n')
    return ProcessResult('passed', 0, 0, 10, False, True)


class WriterCampaignTests(unittest.TestCase):
    def test_profiles_require_fixed_varying_seeds_populations_and_all_scenarios(self):
        for profile, count, seeds, maximum, long in (('ci', 23, 2, 8, 0), ('nightly', 62, 4, 32, 1)):
            with self.subTest(profile=profile):
                plan = generate(profile, 42)
                self.assertEqual(plan, generate(profile, 42))
                self.assertEqual(len(plan['cases']), count)
                self.assertEqual(len(set(plan['seeds'])), seeds)
                self.assertTrue(set(PROFILES[profile]['fixed_seeds']) <= set(plan['seeds']))
                self.assertNotEqual(plan['seeds'], generate(profile, 43)['seeds'])
                # The many-rotation case runs once per nightly campaign, on the
                # first varying seed, with its own case watchdog.
                rotations = [case for case in plan['cases'] if case['scenario'] == 'long-rotations']
                self.assertEqual(len(rotations), long)
                for case in rotations:
                    self.assertEqual((case['seed'], case['clients'], case['steps'], case['timeout_ms']),
                                     (plan['seeds'][len(PROFILES[profile]['fixed_seeds'])], 4, LONG_ROTATIONS, LONG_ROTATION_TIMEOUT_MS))
                # So does the stream-age case, once per ci and nightly campaign.
                ages = [case for case in plan['cases'] if case['scenario'] == 'stream-age']
                self.assertEqual([(case['seed'], case['clients'], case['steps'], case['timeout_ms']) for case in ages],
                                 [(plan['seeds'][len(PROFILES[profile]['fixed_seeds'])], 4, STREAM_AGE_PERIODS[profile],
                                   STREAM_AGE_TIMEOUTS_MS[profile])])
                self.assertTrue(all('timeout_ms' not in case for case in plan['cases']
                                    if case['scenario'] not in ('long-rotations', 'stream-age')))
                for seed in plan['seeds']:
                    cases = [case for case in plan['cases'] if case['seed'] == seed]
                    self.assertEqual({case['scenario'] for case in cases} - {'long-rotations', 'soak', 'stream-age'},
                                     set(SCENARIOS) - {'long-rotations', 'soak', 'stream-age', 'staged-samples',
                                                       'staged-samples-reduction'})
                    self.assertEqual(max(case['clients'] for case in cases), maximum)
                    for case in cases:
                        validate_case(case)
                # Callers cannot mutate the policy through a returned plan.
                plan['budgets']['fixed_seeds'].clear()
                self.assertTrue(generate(profile, 42)['budgets']['fixed_seeds'])

    def test_soak_profile_is_one_long_case_on_a_varying_seed(self):
        plan = generate('soak', 42)
        self.assertEqual(plan, generate('soak', 42))
        self.assertEqual(len(plan['cases']), 1)
        case = plan['cases'][0]
        self.assertEqual((case['id'], case['scenario'], case['clients'], case['steps'], case['timeout_ms']),
                         ('0000-soak-4', 'soak', 4, SOAK_ROTATIONS, SOAK_TIMEOUT_MS))
        self.assertEqual(plan['seeds'], [case['seed']])
        self.assertNotEqual(case['seed'], generate('soak', 43)['cases'][0]['seed'])
        self.assertEqual(OBSERVER_PEAK['soak'], 1)
        # The duration is count x pause; both watchdogs only catch hangs.
        self.assertLess(SOAK_ROTATIONS * SOAK_PAUSE_MS, SOAK_TIMEOUT_MS)
        self.assertLess(SOAK_TIMEOUT_MS, PROFILES['soak']['campaign_timeout_ms'])
        validate_case(case)
        for changed in (dict(case, steps=8), dict(case, timeout_ms=LONG_ROTATION_TIMEOUT_MS)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_case(changed)
        self.assertFalse(any(case['scenario'] == 'soak' for profile in ('ci', 'nightly')
                             for case in generate(profile, 42)['cases']))

    def test_package_profile_runs_the_staged_samples_at_fixed_and_varying_seeds_and_one_reduction_self_test(self):
        from tools.live.writer_campaign import (STAGED_PROFILES, STAGED_REDUCTION_CLIENTS, STAGED_REDUCTION_ROUNDS,
                                                STAGED_REDUCTION_TIMEOUT_MS)
        plan = generate('package', 42)
        self.assertEqual(plan, generate('package', 42))
        self.assertEqual(STAGED_PROFILES, ('package',))
        options = PROFILES['package']
        # The self-test runs once, on the first fixed seed, with its own watchdog.
        self.assertEqual([(case['scenario'], case['clients'], case['steps'], case['seed'], case.get('timeout_ms'))
                          for case in plan['cases']],
                         [('staged-samples', options['graph_clients'], options['sample_rounds'], 42, None),
                          ('staged-samples-reduction', STAGED_REDUCTION_CLIENTS, STAGED_REDUCTION_ROUNDS, 42,
                           STAGED_REDUCTION_TIMEOUT_MS),
                          ('staged-samples', options['graph_clients'], options['sample_rounds'], plan['seeds'][1],
                           None)])
        self.assertLess(STAGED_REDUCTION_TIMEOUT_MS, options['campaign_timeout_ms'])
        # The watchdog is at least twice the 300 s that a windows-x86_64 stock-25.11 runner overran (about 330 s), and
        # the campaign's watchdog never ends a case that is still inside its own.
        self.assertGreaterEqual(STAGED_REDUCTION_TIMEOUT_MS, 2 * 300_000)
        self.assertGreater(options['campaign_timeout_ms'],
                           sum(case.get('timeout_ms', options['case_timeout_ms']) for case in plan['cases']))
        self.assertLessEqual(STAGED_REDUCTION_CLIENTS + OBSERVER_PEAK['staged-samples-reduction'],
                             options['max_client_processes'])
        self.assertEqual(plan['seeds'][0], 42)
        self.assertEqual(len(set(plan['seeds'])), 2)
        self.assertNotEqual(plan['seeds'], generate('package', 43)['seeds'])
        # The two samples, then a fresh observer or the diagnostics sample, fit the client budget.
        self.assertLessEqual(options['graph_clients'] + OBSERVER_PEAK['staged-samples'], options['max_client_processes'])
        for case in plan['cases']:
            validate_case(case)
        # No campaign without a stage runs either case.
        self.assertFalse(any(case['scenario'].startswith('staged-samples') for profile in ('ci', 'nightly', 'soak')
                             for case in generate(profile, 42)['cases']))

    def test_stream_age_cases_cross_many_age_limits_within_their_token_supply(self):
        from tools.live import writer_campaign as campaign
        # The limit is the recorded 240-minute ingress stream timeout; every second period reaches it.
        self.assertEqual((campaign.STREAM_AGE_MAX_MS, 2 * campaign.STREAM_AGE_STEP_MS), (14_400_000, 14_400_000))
        for profile, periods in STREAM_AGE_PERIODS.items():
            with self.subTest(profile=profile):
                scale = campaign.stream_age_scale(periods)
                self.assertTrue(scale['thresholds'])
                self.assertLess(scale['warmup'], periods // 2)
                self.assertEqual(scale['slope'], profile == 'nightly')
                # The tokens cover the first one and one refresh per expiry.
                self.assertGreater(campaign.STREAM_AGE_MAX_GENERATIONS, 1 + periods // 2)
                self.assertLess(STREAM_AGE_TIMEOUTS_MS[profile], PROFILES[profile]['campaign_timeout_ms'])
        self.assertGreaterEqual(campaign.STREAM_AGE_HARNESS_PERIODS // 2, 2)
        self.assertFalse(campaign.STREAM_AGE_HARNESS['thresholds'] or campaign.STREAM_AGE_HARNESS['slope'])

    def test_soak_gauges_cover_every_gauge_the_diagnostic_schema_defines(self):
        from tools.live import writer_campaign as campaign
        schema = json.loads((ROOT / 'tests/contracts/diagnostic-snapshot.schema.json').read_text())
        definitions, fields, gauge_leaves = schema['$defs'], set(), [0]

        def resolve(node):
            while '$ref' in node:
                node = definitions[node['$ref'].rsplit('/', 1)[1]]
            return node

        def example(node, name):
            # One row per table, its dimensions set, so every field the schema defines appears once.
            node = resolve(node)
            if node.get('type') == 'object' and 'properties' in node:
                return {key: example(value, key) for key, value in node['properties'].items()}
            if node.get('type') == 'array':
                row = example(node['items'], name)
                if isinstance(row, dict):
                    row.update(dict.fromkeys(node.get('x-unique-dimensions', ()), 1))
                return [row]
            fields.add(name)
            gauge_leaves[0] += name in campaign.SOAK_GAUGE_FIELDS
            return 0
        metrics = example(definitions['metrics'], 'metrics')
        # Every scalar metric field is classified, and no classification names a field the schema lacks.
        self.assertEqual(campaign.SOAK_GAUGE_FIELDS & campaign.SOAK_NON_GAUGE_FIELDS, set())
        self.assertEqual(campaign.SOAK_GAUGE_FIELDS | campaign.SOAK_NON_GAUGE_FIELDS, fields)
        gauges = campaign.soak_gauges(metrics)
        self.assertEqual(len(gauges), gauge_leaves[0])
        self.assertLessEqual({'rpc_in_flight', 'auth.blocked_endpoints', 'readiness.1.initial_waiting',
                              'readiness.1.recovering', 'subjects.1.current', 'subjects.1.states.1.current',
                              'capacity.pending_bytes.reservations.current', 'capacity.pending_bytes.current_bytes',
                              'capacity.request_slots.timing.current', 'cq_tags.handling', 'jitter.current',
                              'executors.operational.queued', 'executors.diagnostic.running'}, set(gauges))
        with self.assertRaisesRegex(ValueError, 'without a kind or state'):
            campaign.soak_gauges({'table': [{'current': 0}]})

    def test_rss_growth_fits_each_thread_count_and_bounds_each_step(self):
        from unittest import mock
        from tools.live import writer_campaign as campaign
        mib = 1024 * 1024
        # 300 samples of a writer warmed up at 68 threads, as the Windows stream-age case's second half.
        def series(step_at=None, threads=70, step=0, leak=0, leak_from=0):
            points = []
            for index in range(300):
                after = step_at is not None and index >= step_at
                value = 100 * mib + (step if after else 0) + leak * max(0, index - leak_from)
                points.append((300 + index, value, threads if after else 68))
            return points
        with mock.patch.multiple(campaign, SOAK_EXTRA_THREADS=4, SOAK_RSS_THREAD_BYTES=mib):
            flat = campaign.rss_growth(series(), 68)
            self.assertEqual((flat['failures'], [segment['samples'] for segment in flat['segments']]), ([], [300]))
            # A step at a permitted thread increase, with the 64 KiB granule that follows it, passes; the segment
            # after it is too short to fit and is bounded by its growth.
            settled = series(step_at=250, step=int(0.6 * mib))
            settled[-25:] = [(step, value + 65536, threads) for step, value, threads in settled[-25:]]
            stepped = campaign.rss_growth(settled, 68)
            self.assertEqual(stepped['failures'], [])
            self.assertEqual([(s['samples'], s['threads'], s['slope'] is None) for s in stepped['segments']],
                             [(250, 68, False), (50, 70, True)])
            # A steady leak inside a fitted segment fails on its slope.
            leak = campaign.rss_growth(series(leak=campaign.SOAK_RSS_SLOPE_BYTES), 68)
            self.assertEqual(len(leak['failures']), 1)
            self.assertIn('slope', leak['failures'][0])
            # So does a leak in a segment too short to fit, on its growth.
            short = campaign.rss_growth(series(step_at=250, step=int(0.6 * mib), leak=48 * 1024, leak_from=250), 68)
            self.assertEqual(len(short['failures']), 1)
            self.assertIn('growth', short['failures'][0])
            # A step larger than the per-thread allowance of the threads it added fails.
            large = campaign.rss_growth(series(step_at=250, step=2 * mib + 600 * 1024), 68)
            self.assertEqual(len(large['failures']), 1)
            self.assertIn('2 added threads', large['failures'][0])
            # A half that opens inside a burst of two pool threads, which commits 0.3 MiB before they retire, passes
            # (writer 1 of the Windows stream-age case in release-candidate run 40837518792): the burst's threads
            # above the fewest any sample holds count as added. A leak inside such a burst still fails on its growth.
            def opening(leak=0):
                return [(300 + index, 100 * mib + min(index, 60) * (5 * 1024 + leak), 70 if index <= 60 else 68)
                        for index in range(300)]
            burst = campaign.rss_growth(opening(), 71)
            self.assertEqual(burst['failures'], [])
            self.assertEqual([(s['samples'], s['threads']) for s in burst['segments']], [(61, 70), (239, 68)])
            leaking = campaign.rss_growth(opening(leak=48 * 1024), 71)
            self.assertEqual(len(leaking['failures']), 1)
            self.assertIn('2 added threads', leaking['failures'][0])
            # A thread increase beyond SOAK_EXTRA_THREADS fails, whatever the memory does.
            many = campaign.rss_growth(series(step_at=250, threads=73), 68)
            self.assertEqual(len(many['failures']), 50)
            self.assertIn('more than 68 + 4', many['failures'][0])

    def test_soak_thresholds_and_cadence_are_frozen(self):
        from tools.live import writer_campaign as campaign
        # Frozen after the 200-rotation calibration; changing one is a reviewed test change. Windows allows a few
        # on-demand worker threads and twice as many handles, whose count also moves without a thread change.
        extra = 4 if sys.platform == 'win32' else 0
        self.assertEqual((campaign.SOAK_RSS_RATIO, campaign.SOAK_RSS_SLACK_BYTES, campaign.SOAK_RSS_SLOPE_BYTES,
                          campaign.SOAK_RSS_MIN_SEGMENT, campaign.SOAK_EXTRA_THREADS, campaign.SOAK_EXTRA_FILES,
                          campaign.SOAK_RSS_THREAD_BYTES),
                         (1.10, 8 * 1024 * 1024, 2048, 200, extra, 2 * extra, 1024 * 1024 if extra else 0))
        # gRPC's iomgr timer threads only grow, by timing, so they have an absolute bound of their own.
        self.assertEqual((campaign.GRPC_TIMER_THREAD, campaign.SOAK_MAX_GRPC_TIMER_THREADS), ('grpc_global_tim', 4))
        for scale, rotations in ((campaign.soak_scale(), SOAK_ROTATIONS), (campaign.SOAK_HARNESS, 8)):
            with self.subTest(scale=scale):
                # Both expiry kinds occur and the baseline precedes the slope window.
                self.assertGreaterEqual(rotations // scale['expiry_every'], 2)
                self.assertLess(scale['warmup'], rotations // 2)
        self.assertTrue(campaign.soak_scale()['thresholds'])
        self.assertFalse(campaign.SOAK_HARNESS['thresholds'] or campaign.SOAK_HARNESS['ping_hold_every'])
        self.assertEqual(campaign.soak_scale()['pause_ms'], SOAK_PAUSE_MS)
        # Twice the refreshes one run can need: proactive ones (30 s lead), one per expiry event and the first token.
        needed = (SOAK_ROTATIONS * SOAK_PAUSE_MS // (campaign.SOAK_TOKEN_TTL_MS - 30000)
                  + SOAK_ROTATIONS // campaign.SOAK_EXPIRY_EVERY + 1)
        self.assertGreater(campaign.SOAK_MAX_GENERATIONS, 2 * needed)

    def test_case_cannot_change_profile_seed_workload_or_test_selection(self):
        original = generate('ci', 42)['cases'][0]
        for key, value in (('seed', 43), ('clients', 3), ('steps', 2), ('driver', 'scene'),
                           ('test_id', 'different'), ('index', True), ('master_seed', -1), ('schema_version', True)):
            with self.subTest(key=key):
                case = dict(original, **{key: value})
                with self.assertRaises(ValueError): validate_case(case)
        rotations = next(case for case in generate('nightly', 42)['cases'] if case['scenario'] == 'long-rotations')
        validate_case(rotations)
        for case in (dict(original, timeout_ms=LONG_ROTATION_TIMEOUT_MS), dict(rotations, timeout_ms=600000),
                     dict(rotations, timeout_ms=True), dict(rotations, steps=8),
                     {key: value for key, value in rotations.items() if key != 'timeout_ms'}):
            with self.subTest(case=case), self.assertRaises(ValueError): validate_case(case)

    def test_case_watchdog_is_the_case_timeout_or_the_profile_default(self):
        plan = generate('nightly', 42)
        plan['context'] = dict(source='synthetic-context')
        observed = {}
        def timed(command, **kwargs):
            case = json.loads(Path(command[-1]).read_text())['case']
            observed[case['scenario']] = kwargs['timeout_ms']
            return runner(command, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            summary = execute(plan, Path(temporary) / 'campaign', dict(plugin=['plugin'], scene=['scene']), {},
                              runner=timed)
        verify_summary(plan, summary)
        self.assertEqual(observed.pop('long-rotations'), LONG_ROTATION_TIMEOUT_MS)
        self.assertEqual(observed.pop('stream-age'), STREAM_AGE_TIMEOUTS_MS['nightly'])
        self.assertEqual(set(observed.values()), {PROFILES['nightly']['case_timeout_ms']})

    def test_execution_context_is_complete_and_type_sensitive(self):
        context = dict(source='source', plugin={'version': 1}, scene={'version': 2}, reference={'python': 'test'},
                       sanitizer={}, package=None)
        arguments = dict(source='source', candidate={'version': 1}, driver='plugin', reference={'python': 'test'}, sanitizer={})
        verify_context(context, **arguments)
        for key in context:
            changed = dict(context)
            del changed[key]
            with self.subTest(key=key), self.assertRaises(ValueError): verify_context(changed, **arguments)
        # The staged package a package campaign records is part of its identity.
        stage = {'path': '/stage', 'manifest_sha256': 'a' * 64}
        verify_context(dict(context, package=stage), **arguments, package=stage)
        for recorded, actual in ((None, stage), (stage, None), (stage, dict(stage, manifest_sha256='b' * 64))):
            with self.subTest(recorded=recorded, actual=actual), self.assertRaises(ValueError):
                verify_context(dict(context, package=recorded), **arguments, package=actual)
        context['plugin']['version'] = True
        with self.assertRaises(ValueError): verify_context(context, **arguments)

    def test_common_replay_requires_all_dependency_records(self):
        recipe = dict(source_fingerprint='a' * 64, plugin_identity={'version': 1},
                      scene_transport_identity={'version': 1}, reference_identity={'python': 'test'}, sanitizer_runtime={},
                      writer_fuzz_profile='ci', staged_package_identity=None)
        self.assertEqual(replay_context(recipe)['plugin'], {'version': 1})
        for key in set(recipe) - {'writer_fuzz_profile', 'staged_package_identity'}:
            for mode in ('missing', 'null', 'wrong-type'):
                changed = dict(recipe)
                if mode == 'missing': del changed[key]
                else: changed[key] = None if mode == 'null' else []
                with self.subTest(key=key, mode=mode), self.assertRaises(ValueError): replay_context(changed)
        # Only the package campaign records a stage, and it must.
        stage = {'path': '/stage', 'manifest_sha256': 'a' * 64}
        self.assertEqual(replay_context(dict(recipe, writer_fuzz_profile='package',
                                             staged_package_identity=stage))['package'], stage)
        for changed in ({key: value for key, value in recipe.items() if key != 'staged_package_identity'},
                        dict(recipe, staged_package_identity=stage), dict(recipe, writer_fuzz_profile='package'),
                        dict(recipe, writer_fuzz_profile='package', staged_package_identity={}),
                        dict(recipe, writer_fuzz_profile='package', staged_package_identity=[])):
            with self.subTest(changed=changed), self.assertRaises(ValueError): replay_context(changed)

    def run_campaign(self, injected=runner):
        plan = generate('ci', 42)
        plan['context'] = dict(source='synthetic-context')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'campaign'
            result = execute(plan, root, dict(plugin=['plugin'], scene=['scene']), {}, runner=injected)
            self.assertEqual(result, json.loads((root / 'summary.json').read_text()))
            self.assertEqual(plan, json.loads((root / 'plan.json').read_text()))
            return plan, result

    def test_every_required_case_has_fresh_directory_and_completed_evidence(self):
        plan, result = self.run_campaign()
        verify_summary(plan, result)
        self.assertTrue(result['passed'] and result['complete'])
        self.assertEqual(len({r['command'][-1] for r in result['results']}), 23)

    def test_first_failure_stops_campaign_without_retry_or_passing_partial_coverage(self):
        for failure in ('crash', 'timeout', 'cleanup', 'truncated', 'missing', 'skipped', 'empty', 'other-case', 'budget'):
            with self.subTest(failure=failure):
                count = 0
                def injected(command, **kwargs):
                    nonlocal count
                    count += 1
                    result = runner(command, **kwargs)
                    if count != 2:
                        return result
                    destination = kwargs['output'].parent / 'writer-case-result.json'
                    if failure == 'crash': return replace(result, outcome='crashed', returncode=-6)
                    if failure == 'timeout': return replace(result, outcome='timeout')
                    if failure == 'cleanup': return replace(result, cleanup_complete=False)
                    if failure == 'truncated': return replace(result, output_truncated=True)
                    if failure == 'missing': destination.unlink()
                    elif failure == 'budget':
                        # Sparse file exceeds the declared on-disk evidence budget.
                        with (destination.parent / 'oversized').open('wb') as output:
                            output.truncate(PROFILES['ci']['case_artifact_bytes'] + 1)
                    else:
                        record = json.loads(destination.read_text())
                        if failure == 'skipped': record['skipped'] = 1
                        elif failure == 'empty': record['tests_run'] = 0
                        else: record['case']['clients'] += 1
                        write_json(destination, record)
                    return result
                plan, summary = self.run_campaign(injected)
                self.assertEqual(count, 2)
                self.assertFalse(summary['passed'])
                with self.assertRaises(ValueError): verify_summary(plan, summary)

    def test_summary_cannot_omit_duplicate_reorder_or_change_required_cases(self):
        plan, original = self.run_campaign()
        for mode in ('missing', 'duplicate', 'reorder', 'plan', 'cleanup', 'returncode', 'peak'):
            with self.subTest(mode=mode):
                summary = copy.deepcopy(original)
                if mode == 'missing': summary['results'].pop()
                elif mode == 'duplicate': summary['results'][-1] = summary['results'][0]
                elif mode == 'reorder': summary['results'].reverse()
                elif mode == 'plan': summary['plan']['seeds'].pop()
                elif mode == 'cleanup': summary['results'][0]['process']['cleanup_complete'] = False
                elif mode == 'returncode': summary['results'][0]['process']['returncode'] = 1
                else: summary['results'][0]['test']['peak_clients'] = 0
                with self.assertRaises(ValueError): verify_summary(plan, summary)

    def test_client_budget_checked_before_spawn_and_retired_clients_release_capacity(self):
        fixture = SimpleNamespace(writer_process_limit=2, drivers=[SimpleNamespace(process=SimpleNamespace(returncode=None))])
        admit_client(fixture)
        self.assertEqual(fixture.writer_peak_clients, 2)
        fixture.drivers.append(SimpleNamespace(process=SimpleNamespace(returncode=None)))
        with self.assertRaisesRegex(AssertionError, 'budget'): admit_client(fixture)
        fixture.drivers[0].process.returncode = 0
        admit_client(fixture)
        self.assertEqual(fixture.writer_peak_clients, 2)

    def test_actual_generated_workload_cannot_fall_back_to_fixture_defaults(self):
        for name in SCENARIOS:
            profile = dict(soak='soak', **{'staged-samples': 'package',
                                           'staged-samples-reduction': 'package'}).get(name, 'nightly')
            case = next(case for case in generate(profile, 42)['cases'] if case['scenario'] == name)
            original = completed(case)
            self.assertTrue(case_passed(case, original))
            for key, value in (('seed', case['seed'] + 1), ('clients', case['clients'] + 1),
                               ('steps', [case['steps'] - 1] * case['clients'])):
                changed = copy.deepcopy(original)
                changed['workload'][key] = value
                with self.subTest(scenario=name, key=key):
                    self.assertFalse(case_passed(case, changed))
            changed = copy.deepcopy(original)
            changed['peak_clients'] += 1
            self.assertFalse(case_passed(case, changed))
        fixture = SimpleNamespace(writer_process_limit=10)
        observe_workload(fixture, dict(master_seed=7, writers=[[1, 2], [3]]))
        self.assertEqual(fixture.writer_observed_workload, dict(seed=7, clients=2, steps=[2, 1]))

    def test_symlink_artifacts_are_rejected_without_following_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'link').symlink_to(ROOT)
            with self.assertRaises(ValueError): artifact_bytes(root)
