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
    'queued-rotations': ('plugin', 'test_plugin.PluginTests.test_seeded_writers_resume_queued_edits_through_two_backend_rotations'),
    'uncertain-before': ('plugin', 'test_plugin.PluginTests.test_rotation_before_execution_retains_uncertain_and_queued_writer_intents'),
    'uncertain-after': ('plugin', 'test_plugin.PluginTests.test_rotation_after_commit_retains_uncertain_and_queued_writer_intents'),
    'structural': ('scene', 'test_graph_transport.GraphTransportTests.test_seeded_structural_writers_account_deleted_and_overwritten_edits'),
    'samples': ('scene', 'test_graph_transport.GraphTransportTests.test_seeded_sample_writers_account_repeated_and_erased_edits'),
    'long-rotations': ('plugin', 'test_plugin.PluginTests.test_seeded_writers_converge_through_many_backend_rotations'),
    'soak': ('plugin', 'test_plugin.PluginTests.test_seeded_writers_soak_with_refresh_ping_and_bounded_growth'),
}
PROFILES = {
    # Failure watchdogs, sized for the slowest qualified platform, windows-x86_64, where each case runs three to
    # five times as long as on linux-x86_64 (the ci campaign takes about 30 s there, the 8-client nightly structural
    # case about 40 s); passing cases never wait on them.
    'ci': dict(fixed_seeds=[42], varying_seeds=1, root_clients=[2, 8], graph_clients=4, sample_rounds=16,
               case_timeout_ms=60000, campaign_timeout_ms=300000,
               case_artifact_bytes=32 * 1024 * 1024, max_client_processes=10),
    'nightly': dict(fixed_seeds=[0, 42], varying_seeds=2, root_clients=[2, 8, 16, 32], graph_clients=8, sample_rounds=32,
                    case_timeout_ms=300000, campaign_timeout_ms=3000000,
                    case_artifact_bytes=64 * 1024 * 1024, max_client_processes=34),
    'soak': dict(fixed_seeds=[], varying_seeds=1, root_clients=[], graph_clients=0, sample_rounds=0,
                 case_timeout_ms=12_000_000, campaign_timeout_ms=12_300_000,
                 case_artifact_bytes=64 * 1024 * 1024, max_client_processes=6),
}
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
# Upper bound of the least-squares RssAnon slope per rotation over the soak's second half.
SOAK_RSS_SLOPE_BYTES = 2048
# Threads a soak writer may hold above its warm-up count. Windows grows its worker thread pools on demand, so a
# Windows writer may hold a few more, each with its handle; a per-rotation leak still exceeds the bound.
SOAK_EXTRA_THREADS = 4 if sys.platform == 'win32' else 0
# Open files (handles on Windows) a soak writer may hold above its warm-up count.
SOAK_EXTRA_FILES = SOAK_EXTRA_THREADS
# The thread name of gRPC's iomgr timer threads on Linux. gRPC starts another whenever its last waiting timer
# thread runs expired timers and keeps each until gRPC shuts down, so the soak counts them apart from the threads it
# bounds by the warm-up count, and bounds them absolutely.
GRPC_TIMER_THREAD = 'grpc_global_tim'
SOAK_MAX_GRPC_TIMER_THREADS = 4
# Soak rotations between holds that wait in one READY generation for a periodic Ping.
SOAK_PING_HOLD_EVERY = 100
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
    'window_halved', 'window_high_water', 'bounds', 'bytes', 'cause',
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


OBSERVER_PEAK = {'root-fields': 2, 'structural': 1, 'samples': 1, 'queued-rotations': 1,
                 'uncertain-before': 0, 'uncertain-after': 0, 'long-rotations': 1, 'soak': 1}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def generate(profile, master_seed):
    if profile not in PROFILES or type(master_seed) is not int or not 0 <= master_seed < 2**64:
        raise ValueError('writer campaign requires ci/nightly/soak profile and uint64 seed')
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
        workloads = [('root-fields', clients, 16) for clients in options['root_clients']]
        workloads += [('structural', options['graph_clients'], 12),
                      ('samples', options['graph_clients'], options['sample_rounds']),
                      ('queued-rotations', 4, 12), ('uncertain-before', 4, 4), ('uncertain-after', 4, 4)]
        if profile == 'soak':
            workloads = [('soak', 4, SOAK_ROTATIONS)]
        # One long case per nightly campaign; per seed it would exceed the
        # campaign watchdog. It varies with the master seed.
        if profile == 'nightly' and position == len(options['fixed_seeds']):
            workloads.append(('long-rotations', 4, LONG_ROTATIONS))
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
            cases.append(case)
    return dict(schema_version=1, profile=profile, master_seed=master_seed, seeds=seeds,
                budgets=dict(options), cases=cases,
                replay_scope='generated inputs and named fault recipes; OS/gRPC scheduling may differ',
                coverage='bounded root fields, structural/sample intents, safe rotations and uncertain-write recovery')


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


def verify_context(expected, *, source, candidate, driver, reference, sanitizer):
    if not isinstance(expected, dict) or set(expected) != {'source', 'plugin', 'scene', 'reference', 'sanitizer'}:
        raise ValueError('missing writer campaign provenance')
    if canonical([expected['source'], expected[driver], expected['reference'], expected['sanitizer']]) != canonical(
            [source, candidate, reference, sanitizer]):
        raise ValueError('writer campaign execution identity differs')


def replay_context(recipe):
    """Campaign replay requires every recorded dependency, even for empty options."""
    fields = dict(source='source_fingerprint', plugin='plugin_identity', scene='scene_transport_identity',
                  reference='reference_identity', sanitizer='sanitizer_runtime')
    try:
        context = {name: recipe[key] for name, key in fields.items()}
        if (not isinstance(context['source'], str) or len(context['source']) != 64 or
                any(c not in '0123456789abcdef' for c in context['source']) or
                any(not isinstance(context[name], dict) or not context[name] for name in ('plugin', 'scene', 'reference')) or
                not isinstance(context['sanitizer'], dict)):
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
