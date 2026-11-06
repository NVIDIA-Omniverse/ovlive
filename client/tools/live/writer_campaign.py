# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded writer fuzz profiles and completion evidence, independent of clients.

The profile is mandatory coverage, not an arbitrary test selector. Its seed
replays generated inputs and named fault recipes, never an OS thread schedule.
"""
import hashlib
import json
import copy
import sys

SCENARIOS = {
    'root-fields': ('plugin', 'test_plugin.PluginTests.test_seeded_public_writers_account_every_edit_and_conflict'),
    # The coalesced scenarios advertise chained writes and author plain USD change blocks, so each writer chains its
    # writes and coalesces blocks while its window is full; their plain counterparts bind one intent to each command.
    'root-fields-coalesced': ('plugin', 'test_plugin.PluginTests.test_seeded_coalesced_public_writers_account_every_edit_and_conflict'),
    'queued-rotations': ('plugin', 'test_plugin.PluginTests.test_seeded_writers_resume_queued_edits_through_two_backend_rotations'),
    'queued-rotations-coalesced': ('plugin', 'test_plugin.PluginTests.test_seeded_coalesced_writers_resume_queued_edits_through_two_backend_rotations'),
    'uncertain-before': ('plugin', 'test_plugin.PluginTests.test_rotation_before_execution_retains_uncertain_and_queued_writer_intents'),
    'uncertain-after': ('plugin', 'test_plugin.PluginTests.test_rotation_after_commit_retains_uncertain_and_queued_writer_intents'),
    'structural': ('scene', 'test_graph_transport.GraphTransportTests.test_seeded_structural_writers_account_deleted_and_overwritten_edits'),
    'samples': ('scene', 'test_graph_transport.GraphTransportTests.test_seeded_sample_writers_account_repeated_and_erased_edits'),
    # Writers author under shared ancestors the layer lacks, so their implicit containers replace each other.
    'first-create': ('scene', 'test_graph_transport.GraphTransportTests.test_seeded_first_create_writers_keep_every_writers_content'),
    'long-rotations': ('plugin', 'test_plugin.PluginTests.test_seeded_writers_converge_through_many_backend_rotations'),
    'soak': ('plugin', 'test_plugin.PluginTests.test_seeded_writers_soak_with_refresh_ping_and_bounded_growth'),
    'stream-age': ('plugin', 'test_plugin.PluginTests.test_seeded_writers_and_joins_survive_stream_age_expiries_after_server_token_expiry'),
    # The samples of a stage tools/live/build.py produced, run as its users run them, beside seeded plugin writers.
    'staged-samples': ('plugin', 'test_staged_samples.StagedSampleTests.test_staged_samples_converge_beside_seeded_writers'),
    # The staged-samples reduction self-test: a named test-only defect reduced, replayed and replayed standalone.
    'staged-samples-reduction': ('plugin', 'test_staged_samples.StagedSampleTests.'
                                           'test_staged_samples_corrupt_reduction_reexecutes_staged_samples_and_service'),
}
PROFILES = {
    # Failure watchdogs, sized for the slowest qualified platform, windows-x86_64, where each case runs three to
    # five times as long as on linux-x86_64 (the ci campaign takes about 30 s there, the 8-client nightly structural
    # case about 40 s); passing cases never wait on them.
    'ci': dict(fixed_seeds=[42], varying_seeds=1, root_clients=[2, 8], graph_clients=4, sample_rounds=16,
               first_create_rounds=6,
               case_timeout_ms=60000, campaign_timeout_ms=300000,
               case_artifact_bytes=32 * 1024 * 1024, max_client_processes=10),
    'nightly': dict(fixed_seeds=[0, 42], varying_seeds=2, root_clients=[2, 8, 16, 32], graph_clients=8, sample_rounds=32,
                    first_create_rounds=12,
                    case_timeout_ms=300000, campaign_timeout_ms=3000000,
                    case_artifact_bytes=64 * 1024 * 1024, max_client_processes=34),
    'soak': dict(fixed_seeds=[], varying_seeds=1, root_clients=[], graph_clients=0, sample_rounds=0, first_create_rounds=0,
                 case_timeout_ms=12_000_000, campaign_timeout_ms=12_300_000,
                 case_artifact_bytes=64 * 1024 * 1024, max_client_processes=6),
    # The staged-package campaign: only the CI profiles that stage the package run it, after staging, with
    # graph_clients seeded writers and sample_rounds rounds per staged-samples case. Its watchdog exceeds the sum of
    # its cases' watchdogs, so a case still inside its own is never ended by the campaign's.
    'package': dict(fixed_seeds=[42], varying_seeds=1, root_clients=[], graph_clients=4, sample_rounds=8, first_create_rounds=0,
                    case_timeout_ms=120000, campaign_timeout_ms=900000,
                    case_artifact_bytes=32 * 1024 * 1024, max_client_processes=8),
}
# The writer campaign profiles that run a staged package's samples, and so need its stage.
STAGED_PROFILES = ('package',)
# The package campaign's one staged-samples reduction self-test, on its first fixed seed: its writers and rounds, and
# its own process watchdog, which covers every trial and standalone replay. The failing trial, the reduction's trials,
# the minimal plan's replay, two trials without the defect and three standalone replays take 240 to 330 s on the
# windows-x86_64 runners; the watchdog allows about twice that.
STAGED_REDUCTION_CLIENTS = 2
STAGED_REDUCTION_ROUNDS = 4
STAGED_REDUCTION_TIMEOUT_MS = 600_000
# Event-driven backend rotations per layer in the one nightly long-rotations
# case, and that case's own process watchdog.
LONG_ROTATIONS = 500
LONG_ROTATION_TIMEOUT_MS = 900000
# Backend rotations in the one soak case; its duration is this count times the pause.
SOAK_ROTATIONS = 600
# READY time after each soak rotation: load (token lifetime, holdback of paced peer edits), never synchronization.
SOAK_PAUSE_MS = 5000
# A server-side expiry of every writer's current token every this many soak rotations.
SOAK_EXPIRY_EVERY = 100
# Client-side lifetime of each soak writer token; the runtime refreshes it proactively before it ends.
SOAK_TOKEN_TTL_MS = 180_000
# Independent peer commits spread across each soak pause.
SOAK_PEER_EDITS = 2
# Soak rotations before the resource baseline: the first later-or-equal rotation without an expiry event.
SOAK_WARMUP = 100
# Tokens issued per soak writer; the last rotation's generation stays below it.
SOAK_MAX_GENERATIONS = 256
# The soak case's own process watchdog.
SOAK_TIMEOUT_MS = 12_000_000
# Final soak RssAnon and VmHWM bound: SOAK_RSS_RATIO times the warm-up value plus SOAK_RSS_SLACK_BYTES.
SOAK_RSS_RATIO = 1.10
# Absolute slack of the final soak RSS bound.
SOAK_RSS_SLACK_BYTES = 8 * 1024 * 1024
# Upper bound of the least-squares RssAnon slope per rotation over the soak's second half, fitted on its own to each
# run of samples with an unchanged thread count (rss_growth).
SOAK_RSS_SLOPE_BYTES = 2048
# Fewest samples such a run needs to be fitted on its own: the 200 rotations the slope bound was calibrated on. Memory
# is committed in 64 KiB granules, and one granule inside a 50-sample run already fits as about 2 KiB per sample, so a
# shorter run is bounded by its total growth instead (rss_growth).
SOAK_RSS_MIN_SEGMENT = 200
# Threads a soak writer may hold above its warm-up count. Windows grows its worker thread pools on demand, so a
# Windows writer may hold a few more, each with its handle; a per-rotation leak still exceeds the bound.
SOAK_EXTRA_THREADS = 4 if sys.platform == 'win32' else 0
# Open files (handles on Windows) a soak writer may hold above its warm-up count. A Windows writer's handle count also
# moves by a few between samples with no change in its threads (up to 5 above a one-period warm-up in release-candidate
# run 40790418370), so Windows allows twice the thread allowance; a per-period leak still exceeds it over a campaign.
SOAK_EXTRA_FILES = 2 * SOAK_EXTRA_THREADS
# RssAnon (private commit on Windows) a writer may gain for each thread it adds within SOAK_EXTRA_THREADS. When two
# on-demand pool threads start, a Windows writer's private commit steps up 0.16 to 0.74 MiB, the 64 KiB granules that
# follow within 25 periods included (stream-age case, release-candidate runs 40790418370, 40804491344 and
# 40810618109), so 1 MiB per thread leaves headroom; elsewhere a writer adds no thread.
SOAK_RSS_THREAD_BYTES = 1024 * 1024 if sys.platform == 'win32' else 0
# The thread name of gRPC's iomgr timer threads on Linux. gRPC starts another whenever its last waiting timer
# thread runs expired timers and keeps each until gRPC shuts down, so the soak counts them apart from the threads it
# bounds by the warm-up count, and bounds them absolutely.
GRPC_TIMER_THREAD = 'grpc_global_tim'
SOAK_MAX_GRPC_TIMER_THREADS = 4
# Soak rotations between holds that wait in one READY generation for a periodic Ping.
SOAK_PING_HOLD_EVERY = 100
# The stream-age cases' maximum stream age, the 240-minute ingress stream timeout the deployment research records,
# and the stream clock step of each period: half the limit, so every second period ends every read and join stream.
STREAM_AGE_MAX_MS = 240 * 60 * 1000
STREAM_AGE_STEP_MS = STREAM_AGE_MAX_MS // 2
# Periods of the ci and nightly stream-age cases, half of them expiries, and each case's own process watchdog.
STREAM_AGE_CI_PERIODS = 100
STREAM_AGE_NIGHTLY_PERIODS = 600
STREAM_AGE_CI_TIMEOUT_MS = 120_000
STREAM_AGE_NIGHTLY_TIMEOUT_MS = 900_000
# Periods of plain plugin-reference discovery: two expiries.
STREAM_AGE_HARNESS_PERIODS = 4
# Tokens issued per stream-age writer; the last period's generation stays below it.
STREAM_AGE_MAX_GENERATIONS = 2048
# Independent peer commits per stream-age period.
STREAM_AGE_PEER_EDITS = 1
# Plain discovery takes the baseline after the first period and asserts no memory bound.
STREAM_AGE_HARNESS = dict(warmup=1, thresholds=False, slope=False)


STREAM_AGE_PERIODS = {'ci': STREAM_AGE_CI_PERIODS, 'nightly': STREAM_AGE_NIGHTLY_PERIODS}
STREAM_AGE_TIMEOUTS_MS = {'ci': STREAM_AGE_CI_TIMEOUT_MS, 'nightly': STREAM_AGE_NIGHTLY_TIMEOUT_MS}


def stream_age_scale(periods):
    """A stream-age campaign case's warm-up and memory assertions: the RSS bounds always, the slope over the
    second half only from 200 periods, so the slope window holds enough periods to separate growth from noise."""
    return dict(warmup=max(2, periods // 10), thresholds=True, slope=periods >= 200)


# Plain plugin-reference discovery runs the soak at harness scale: a short pause,
# both expiry events twice and a baseline within its 8 rotations, no hold and no
# memory thresholds.
SOAK_HARNESS = dict(pause_ms=100, expiry_every=2, warmup=3, ping_hold_every=0, thresholds=False)


def soak_scale():
    """The soak campaign case's pause, expiry, warm-up and hold cadence, with memory thresholds."""
    return dict(pause_ms=SOAK_PAUSE_MS, expiry_every=SOAK_EXPIRY_EVERY, warmup=SOAK_WARMUP,
                ping_hold_every=SOAK_PING_HOLD_EVERY, thresholds=True)


# Scalar metric fields of a diagnostic snapshot that are levels: each returns to
# its idle value once the work it counts is released, so the soak compares every
# one against its warm-up baseline. Every other scalar metric field is a
# cumulative count, a maximum, a table dimension, a configuration or a histogram
# part. test_writer_campaign classifies every field of
# tests/contracts/diagnostic-snapshot.schema.json into exactly one of the two.
SOAK_GAUGE_FIELDS = frozenset({
    'current', 'current_bytes', 'current_bytes_overflow', 'rpc_in_flight', 'blocked_endpoints', 'initial_waiting',
    'recovering', 'outstanding', 'handling', 'retained', 'queued', 'running', 'live_workers', 'oldest_queued_age_ms',
    'oldest_running_age_ms', 'in_flight'})
SOAK_NON_GAUGE_FIELDS = frozenset({
    'abandoned', 'admitted', 'admitted_bytes', 'admitted_updates', 'blocked_endpoints_high_water', 'chain_resends',
    'coalesced_turns', 'in_flight_high_water', 'max_turns_per_update', 'resent_writes', 'window_grown',
    'window_halved', 'window_high_water', 'bounds', 'bytes', 'call', 'cause',
    'closed_incidents_evicted', 'completed', 'configured_workers', 'count', 'counts', 'detached', 'emitted', 'enqueued',
    'event_code', 'frozen_generation_starts', 'high_water', 'high_water_bytes', 'high_water_bytes_overflow',
    'history_dropped', 'history_overwritten', 'invalid', 'kind', 'operation', 'outstanding_high_water', 'phase',
    'queue_high_water', 'reason', 'receive_to_apply_max_ms', 'received_not_ok', 'received_ok', 'rejected',
    'rejected_bytes', 'released', 'released_bytes', 'resolution', 'result', 'retained_high_water', 'retired',
    'rpc_attempts', 'rpc_completions', 'scope', 'sink_dropped', 'sink_failures', 'sinks_disabled', 'stale_replacements',
    'started', 'state', 'submitted', 'sum_ms', 'suppressed', 'unsampled_starts', 'warning_suppressed'})
# A metric table row that holds a gauge is named by its kind or state.
SOAK_ROW_KEYS = ('kind', 'state')


def soak_gauges(metrics):
    """Every gauge of one diagnostic snapshot's metrics, as {dotted path: value}.

    A table row's path part is its kind or state, as in subjects.3.states.2.current.
    """
    gauges = {}

    def walk(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    walk(item, path + (key,))
                elif key in SOAK_GAUGE_FIELDS:
                    gauges['.'.join(path + (key,))] = item
        elif isinstance(value, list):
            for row in value:
                label = next((str(row[key]) for key in SOAK_ROW_KEYS if isinstance(row, dict) and key in row), None)
                before = len(gauges)
                walk(row, path + (label or '-',))
                if label is None and len(gauges) != before:
                    raise ValueError('a metric row holds a gauge without a kind or state: ' + '.'.join(path))
    walk(metrics, ())
    return gauges


OBSERVER_PEAK = {'root-fields': 2, 'root-fields-coalesced': 2, 'structural': 1, 'samples': 1, 'first-create': 1,
                 'queued-rotations': 1,
                 'queued-rotations-coalesced': 1,
                 'uncertain-before': 0, 'uncertain-after': 0, 'long-rotations': 1, 'soak': 1, 'stream-age': 1,
                 # The two participant samples and, one at a time, a fresh observer or the diagnostics sample.
                 'staged-samples': 3,
                 # The same peak, in the self-test's trial of the whole plan without its defect.
                 'staged-samples-reduction': 3}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def generate(profile, master_seed):
    if profile not in PROFILES or type(master_seed) is not int or not 0 <= master_seed < 2**64:
        raise ValueError('writer campaign requires ci/nightly/soak/package profile and uint64 seed')
    options = copy.deepcopy(PROFILES[profile])
    seeds = list(options['fixed_seeds'])
    ordinal = 0
    while len(seeds) < len(options['fixed_seeds']) + options['varying_seeds']:
        value = int.from_bytes(hashlib.sha256(f'writer-campaign-v1:{master_seed}:{ordinal}'.encode()).digest()[:8], 'big')
        ordinal += 1
        if value not in seeds:
            seeds.append(value)
    cases = []
    for position, seed in enumerate(seeds):
        workloads = [(scenario, clients, 16) for clients in options['root_clients']
                     for scenario in ('root-fields', 'root-fields-coalesced')]
        workloads += [('structural', options['graph_clients'], 12),
                      ('samples', options['graph_clients'], options['sample_rounds']),
                      ('first-create', options['graph_clients'], options['first_create_rounds']),
                      ('queued-rotations', 4, 12), ('queued-rotations-coalesced', 4, 12),
                      ('uncertain-before', 4, 4), ('uncertain-after', 4, 4)]
        if profile == 'soak':
            workloads = [('soak', 4, SOAK_ROTATIONS)]
        if profile == 'package':
            workloads = [('staged-samples', options['graph_clients'], options['sample_rounds'])]
            if position == 0:
                workloads.append(('staged-samples-reduction', STAGED_REDUCTION_CLIENTS, STAGED_REDUCTION_ROUNDS))
        # One long case per nightly campaign; per seed it would exceed the
        # campaign watchdog. It varies with the master seed.
        if profile == 'nightly' and position == len(options['fixed_seeds']):
            workloads.append(('long-rotations', 4, LONG_ROTATIONS))
        # One stream-age case per ci and nightly campaign, on the first varying seed.
        if profile in STREAM_AGE_PERIODS and position == len(options['fixed_seeds']):
            workloads.append(('stream-age', 4, STREAM_AGE_PERIODS[profile]))
        for scenario, clients, steps in workloads:
            index = len(cases)
            case = dict(schema_version=1, profile=profile, master_seed=master_seed,
                        index=index, id=f'{index:04d}-{scenario}-{clients}', seed=seed,
                        scenario=scenario, clients=clients, steps=steps,
                        driver=SCENARIOS[scenario][0], test_id=SCENARIOS[scenario][1])
            if scenario == 'long-rotations':
                case['timeout_ms'] = LONG_ROTATION_TIMEOUT_MS
            if scenario == 'soak':
                case['timeout_ms'] = SOAK_TIMEOUT_MS
            if scenario == 'stream-age':
                case['timeout_ms'] = STREAM_AGE_TIMEOUTS_MS[profile]
            if scenario == 'staged-samples-reduction':
                case['timeout_ms'] = STAGED_REDUCTION_TIMEOUT_MS
            cases.append(case)
    return dict(schema_version=1, profile=profile, master_seed=master_seed, seeds=seeds,
                budgets=dict(options), cases=cases,
                replay_scope='generated inputs and named fault recipes; OS/gRPC scheduling may differ',
                coverage='bounded root fields, structural/sample intents, first creates under shared missing ancestors, '
                         'safe rotations, uncertain-write recovery and staged-package samples')


def validate_case(case):
    try:
        campaign = generate(case['profile'], case['master_seed'])
        index = case['index']
        if type(index) is not int or not 0 <= index < len(campaign['cases']):
            raise ValueError('invalid writer case index')
        if canonical(case) != canonical(campaign['cases'][index]):
            raise ValueError('writer case differs from mandatory profile')
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('invalid writer case') from error
    return case


def verify_context(expected, *, source, candidate, driver, reference, sanitizer, package=None):
    """`package` is the staged package's identity, which only a package campaign records; None elsewhere."""
    if not isinstance(expected, dict) or set(expected) != {'source', 'plugin', 'scene', 'reference', 'sanitizer',
                                                           'package'}:
        raise ValueError('missing writer campaign provenance')
    if canonical([expected['source'], expected[driver], expected['reference'], expected['sanitizer'],
                  expected['package']]) != canonical([source, candidate, reference, sanitizer, package]):
        raise ValueError('writer campaign execution identity differs')


def replay_context(recipe):
    """Campaign replay requires every recorded dependency, even for empty options."""
    fields = dict(source='source_fingerprint', plugin='plugin_identity', scene='scene_transport_identity',
                  reference='reference_identity', sanitizer='sanitizer_runtime', package='staged_package_identity')
    try:
        context = {name: recipe[key] for name, key in fields.items()}
        if (not isinstance(context['source'], str) or len(context['source']) != 64 or
                any(c not in '0123456789abcdef' for c in context['source']) or
                any(not isinstance(context[name], dict) or not context[name] for name in ('plugin', 'scene', 'reference')) or
                not isinstance(context['sanitizer'], dict) or
                (context['package'] is not None and (not isinstance(context['package'], dict) or not context['package'])) or
                (context['package'] is None) != (recipe['writer_fuzz_profile'] not in STAGED_PROFILES)):
            raise ValueError('invalid writer campaign replay identities')
    except (KeyError, TypeError) as error:
        raise ValueError('missing writer campaign replay identities') from error
    return context


def case_passed(case, result):
    """A framework's zero exit or empty selection cannot satisfy a case."""
    return (isinstance(result, dict) and canonical(result.get('case')) == canonical(case) and
            type(result.get('schema_version')) is int and result['schema_version'] == 1 and type(result.get('tests_run')) is int and
            result['tests_run'] == 1 and result.get('passed') is True and
            type(result.get('peak_clients')) is int and
            result['peak_clients'] == case['clients'] + OBSERVER_PEAK[case['scenario']] and
            result['peak_clients'] <= PROFILES[case['profile']]['max_client_processes'] and
            canonical(result.get('workload')) == canonical(dict(seed=case['seed'], clients=case['clients'],
                                                                 steps=[case['steps']] * case['clients'])) and
            all(type(result.get(key)) is int and result[key] == 0 for key in
                ('skipped', 'errors', 'failures', 'unexpected_successes', 'expected_failures')))


def verify_summary(plan, summary):
    if (not isinstance(summary, dict) or type(summary.get('schema_version')) is not int or summary['schema_version'] != 1 or
            summary.get('complete') is not True or summary.get('passed') is not True or
            canonical(summary.get('plan')) != canonical(plan) or
            not isinstance(summary.get('results'), list) or len(summary['results']) != len(plan['cases'])):
        raise ValueError('writer campaign coverage or completion evidence missing')
    for case, result in zip(plan['cases'], summary['results']):
        if not isinstance(result, dict) or not isinstance(result.get('process'), dict):
            raise ValueError('invalid writer campaign process evidence')
        process = result.get('process', {})
        if (result.get('id') != case['id'] or result.get('passed') is not True or
                process.get('outcome') != 'passed' or process.get('cleanup_complete') is not True or
                type(process.get('returncode')) is not int or process['returncode'] != 0 or
                process.get('output_truncated') is not False or not case_passed(case, result.get('test')) or
                type(result.get('artifact_bytes')) is not int or
                not 0 <= result['artifact_bytes'] <= plan['budgets']['case_artifact_bytes']):
            raise ValueError('writer campaign case failed, truncated, incomplete or over budget')
    return summary


def admit_client(fixture):
    """Check campaign client budget before spawning; ordinary tests keep defaults."""
    if not hasattr(fixture, 'writer_process_limit'):
        return
    active = sum(driver.process.returncode is None for driver in fixture.drivers)
    if active >= fixture.writer_process_limit:
        raise AssertionError('writer campaign client process budget exhausted')
    fixture.writer_peak_clients = max(getattr(fixture, 'writer_peak_clients', 0), active + 1)


def observe_workload(fixture, plan):
    """Record actual generated inputs, independently of requested case parameters."""
    if hasattr(fixture, 'writer_process_limit'):
        fixture.writer_observed_workload = dict(seed=plan['master_seed'], clients=len(plan['writers']),
                                                steps=[len(writer) for writer in plan['writers']])


def rss_slope(points):
    """Least-squares slope of (step, bytes) points, in bytes per step."""
    count = len(points)
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    spread = sum((x - mean_x) ** 2 for x, _ in points)
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / spread if spread else 0.0


def rss_growth(points, base_threads):
    """A writer's RssAnon growth over `points`, its (step, bytes, threads) samples in step order, against steady-leak
    bounds: {'segments': [...], 'failures': [...]}, passing when failures is empty.

    The samples split into segments of an unchanged thread count. A segment of at least SOAK_RSS_MIN_SEGMENT samples
    has its own least-squares slope, below SOAK_RSS_SLOPE_BYTES. Every other segment and every thread-count change is
    bounded by its growth: from the last sample before the segment (its first sample for the first segment) to the
    segment's last sample, or to its first sample when the segment is fitted, at most SOAK_RSS_THREAD_BYTES per thread
    the change added plus SOAK_RSS_SLOPE_BYTES per step spanned. The first segment has no change before it, so its
    threads above the fewest any sample holds count as added: `points` can open inside a burst of on-demand pool
    threads whose memory the segment then commits. No sample holds more than SOAK_EXTRA_THREADS threads above
    `base_threads`, the warm-up count."""
    failures, segments = [], []
    fewest = min((threads for _, _, threads in points), default=0)
    for step, _, threads in points:
        if threads > base_threads + SOAK_EXTRA_THREADS:
            failures.append(f'{threads} threads at step {step}, more than {base_threads} + {SOAK_EXTRA_THREADS}')
    runs = []
    for point in points:
        if runs and runs[-1][-1][2] == point[2]:
            runs[-1].append(point)
        else:
            runs.append([point])
    previous = None
    for run in runs:
        start, end = run[0], run[-1]
        fitted = len(run) >= SOAK_RSS_MIN_SEGMENT
        slope = rss_slope([(step, value) for step, value, _ in run]) if fitted else None
        if fitted and slope >= SOAK_RSS_SLOPE_BYTES:
            failures.append(f'slope {slope:.0f} B/step from step {start[0]} to {end[0]}, '
                            f'not below {SOAK_RSS_SLOPE_BYTES}')
        origin = previous or start
        until = start if fitted else end
        added = max(0, start[2] - (origin[2] if previous else fewest))
        growth = until[1] - origin[1]
        limit = added * SOAK_RSS_THREAD_BYTES + SOAK_RSS_SLOPE_BYTES * (until[0] - origin[0])
        if growth > limit:
            failures.append(f'growth {growth} B from step {origin[0]} to {until[0]} with {added} added threads, '
                            f'more than {limit}')
        segments.append(dict(start=start[0], end=end[0], samples=len(run), threads=start[2], slope=slope,
                             growth=growth))
        previous = end
    return dict(segments=segments, failures=failures)

