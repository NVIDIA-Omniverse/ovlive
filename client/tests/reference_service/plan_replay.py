# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone replay of a recorded reduced writer plan of a graph, rotation, period or staged-samples case.

A reduction of these classes records, in its `reduction.json`, a replay context: the case `kind`, its reference
`lane`, the case `scale` a trial needs and the identities every trial ran against (source fingerprint, native
candidate with its SDK manifest, reference-service dependencies and interpreter, sanitizer runtime and, for a
staged-samples case, the stage with the SDK interpreter it records and its sample executables). A replay loads
`best.plan.json` or `original.plan.json` beside it, requires the plan's hash to equal the one the reduction recorded
for that file and the identities to equal the current ones, and only then runs the plan once through the class's
reduction evaluator: a fresh fixture, service and set of native writers. A plan that carries a named test-only
defect replays with it unless the caller removes it explicitly. OS scheduling is not replayed.
"""
import json
import os
from pathlib import Path

import period_plans
import rotation_history
import sample_history
import staged_sample_history
import structural_history
from writer_history import read_json
from writer_reduction import digest

# Exit status of a replay: the recorded signature reproduced, a valid replay that did not reproduce it, and a replay
# refused before any trial (an invalid or altered plan or recipe, a different identity) or without a trial result.
REPRODUCED, NOT_REPRODUCED, REFUSED = 1, 2, 3
# Each replayable class: its reference lane, plan module and plan scenario.
KINDS = {
    'structural': ('scene-reference', structural_history, structural_history.SCENARIO),
    'samples': ('scene-reference', sample_history, 'private-attachment.samples-v1'),
    'queued-rotations': ('plugin-reference', rotation_history, rotation_history.QUEUED),
    'queued-rotations-coalesced': ('plugin-reference', rotation_history, rotation_history.QUEUED),
    'uncertain-before': ('plugin-reference', rotation_history, rotation_history.UNCERTAIN),
    'uncertain-after': ('plugin-reference', rotation_history, rotation_history.UNCERTAIN),
    'long-rotations': ('plugin-reference', period_plans, period_plans.LONG_ROTATIONS),
    'stream-age': ('plugin-reference', period_plans, period_plans.STREAM_AGE),
    'staged-samples': ('plugin-reference', staged_sample_history, staged_sample_history.SCENARIO),
}
# The reduction's recorded hash of each replayable plan file.
PLANS = {'best.plan.json': 'best_sha256', 'original.plan.json': 'original_sha256'}
IDENTITIES = ('source_fingerprint', 'candidate', 'reference_identity', 'sanitizer_runtime')
# The classes whose trials run a staged package's samples, which also record the stage's identity.
STAGED_KINDS = ('staged-samples',)


def identities(kind):
    """The identities a reduction of `kind` records: IDENTITIES, and the stage for a staged-samples case."""
    return IDENTITIES + (('staged_package',) if kind in STAGED_KINDS else ())


def staged_identity(path):
    """The identity of the stage at `path` that a staged-samples trial runs: its absolute path and manifest hash
    (tools/live/build.py stage_identity), the SDK interpreter the stage records, which runs the Python sample, and its
    sample executables and Python sample module, each with its SHA-256. Validating the stage recomputes every member's
    hash and the interpreter's, so each is the current file's."""
    from tools.live import build, platforms
    platform = platforms.host().name
    identity = build.stage_identity(path, platform)
    record = json.loads((Path(identity['path']) / build.MANIFEST).read_text())
    interpreter = record['support_identity']['tools']['python']
    return dict(identity, interpreter=dict(path=interpreter['path'], sha256=interpreter['sha256']),
                samples={name: record['files'][name]
                         for name in (*build.samples(platform), build.PYTHON_PACKAGE + 'sample.py')})


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def context(kind, scale=None, environment=os.environ):
    """The replay context of a reduction of `kind` that this reference lane runs, at the case `scale`.

    The identities are the ones the common runner verified for the lane and passed in its environment, the candidate
    is the manifest of the lane's native driver and a staged-samples case's stage is the one the lane names. A lane
    run outside the common runner lacks them: each missing identity is recorded as None, so the reduction still runs
    and its record is refused for replay.
    """
    from tools.live.sanitizer import recorded
    lane = KINDS[kind][0]
    driver = environment.get('LIVE_TEST_PLUGIN_DRIVER' if lane == 'plugin-reference' else 'LIVE_TEST_SCENE_DRIVER')
    try:
        candidate = json.loads(Path(driver + '.manifest.json').read_text()) if driver else None
    except (OSError, ValueError):
        candidate = None
    reference = environment.get('LIVE_TEST_REFERENCE_IDENTITY')
    recorded_context = dict(kind=kind, lane=lane, scale=scale,
                            source_fingerprint=environment.get('LIVE_TEST_SOURCE_FINGERPRINT'), candidate=candidate,
                            reference_identity=None if reference is None else json.loads(reference),
                            sanitizer_runtime=recorded(environment))
    if kind in STAGED_KINDS:
        stage = environment.get('LIVE_TEST_STAGED_PACKAGE')
        try:
            recorded_context['staged_package'] = staged_identity(stage) if stage else None
        except (OSError, ValueError, KeyError, TypeError):
            recorded_context['staged_package'] = None
    return recorded_context


def valid_scale(kind, scale):
    """Whether `scale` is a case scale a trial of `kind` runs at: None, or for `stream-age` its warm-up period count
    and memory-bound switches (writer_campaign.stream_age_scale)."""
    if scale is None:
        return True
    return (kind == 'stream-age' and isinstance(scale, dict) and set(scale) == {'warmup', 'thresholds', 'slope'} and
            type(scale['warmup']) is int and scale['warmup'] >= 1 and type(scale['thresholds']) is bool and
            type(scale['slope']) is bool)


def load(path, *, without_defect=False):
    """The replay recipe of the reduced plan at `path`, `best.plan.json` or `original.plan.json` beside its
    `reduction.json`; raises ValueError for any other file, an altered plan or hash, or a reduction without a replay
    context.

    With `without_defect`, the plan's named test-only defect is removed after its hash is checked; a plan without one
    is refused.
    """
    path = Path(path)
    if path.name not in PLANS:
        raise ValueError('a reduced plan replay takes best.plan.json or original.plan.json')
    summary = read_json(path.parent / 'reduction.json', limit=16 * 1024 * 1024)
    if not isinstance(summary, dict):
        raise ValueError('invalid reduction record')
    recorded = summary.get('context')
    if (summary.get('schema_version') != 2 or not isinstance(recorded, dict) or recorded.get('kind') not in KINDS or
            set(recorded) != {'kind', 'lane', 'scale', *identities(recorded['kind'])} or
            recorded['lane'] != KINDS[recorded['kind']][0] or not isinstance(summary.get('target'), dict)):
        raise ValueError('reduction records no replay context of a graph, rotation, period or staged-samples writer '
                         'case')
    if not valid_scale(recorded['kind'], recorded['scale']):
        raise ValueError('invalid reduction case scale')
    if any(recorded[key] is None for key in identities(recorded['kind'])):
        raise ValueError('reduction recorded no identity of the lane it ran in, so it cannot be replayed')
    plan = read_json(path, limit=16 * 1024 * 1024)
    if digest(plan) != summary.get(PLANS[path.name]):
        raise ValueError(f'reduced plan differs from the hash its reduction recorded for {path.name}')
    _, module, scenario = KINDS[recorded['kind']]
    if not isinstance(plan, dict) or plan.get('scenario') != scenario:
        raise ValueError(f"reduced plan is not a {recorded['kind']} plan")
    module.validate(plan)
    if without_defect:
        if 'test_service_fault' not in plan:
            raise ValueError('reduced plan carries no named test-only defect')
        plan = {key: value for key, value in plan.items() if key != 'test_service_fault'}
        module.validate(plan)
    return dict(recorded, name=path.name, plan=plan, plan_sha256=digest(plan), target=summary['target'],
                without_test_defect=without_defect)


def verify(recipe, *, source, candidate, reference, sanitizer, staged=None):
    """Require the recipe's recorded identities to equal the current ones before any trial launches. `staged` is the
    current identity of the stage a staged-samples replay names (staged_identity), and None for other classes."""
    actual = dict(source_fingerprint=source, candidate=candidate, reference_identity=reference,
                  sanitizer_runtime=sanitizer)
    if canonical({key: recipe[key] for key in IDENTITIES}) != canonical(actual):
        raise ValueError('reduced plan replay source, candidate, dependencies, interpreter or sanitizer identity '
                         'differs')
    if recipe['kind'] in STAGED_KINDS and canonical(recipe['staged_package']) != canonical(staged):
        raise ValueError('reduced plan replay stage, its SDK interpreter or its sample executables differ')
    if recipe['kind'] not in STAGED_KINDS and staged is not None:
        raise ValueError(f"a reduced {recipe['kind']} plan runs no staged package")
    return recipe


async def replay(recipe, directory):
    """Run the recipe's plan once through its class's reduction evaluator in `directory`; returns the trial's result,
    as the reducer records it."""
    kind, plan = recipe['kind'], recipe['plan']
    if kind in STAGED_KINDS:
        from test_staged_samples import StagedSampleTests, CASE_TEST
        return await StagedSampleTests(CASE_TEST).staged_trial(plan, directory)
    if recipe['lane'] == 'scene-reference':
        from test_graph_transport import GraphTransportTests, GRAPH_WRITER_TESTS
        return await GraphTransportTests(GRAPH_WRITER_TESTS[kind]).graph_trial(kind, plan, directory)
    from test_plugin import PluginTests, PERIOD_TESTS, ROTATION_TESTS
    if kind in ROTATION_TESTS:
        return await PluginTests(ROTATION_TESTS[kind]).rotation_trial(kind, plan, directory)
    case = PluginTests(PERIOD_TESTS[kind])
    if recipe['scale'] is not None:
        case.writer_stream_age = recipe['scale']
    return await case.period_trial(kind, plan, directory)


def reproduced(result, target):
    """Whether a trial result raised exactly the recorded target signature."""
    return (isinstance(result, dict) and result.get('outcome') == 'mismatch' and
            result.get('cleanup_complete') is True and isinstance(result.get('signature'), dict) and
            canonical(result['signature']) == canonical(target))


async def standalone(path, artifacts, *options, environment=os.environ):
    """Run the standalone replay command on the reduced plan `path` from inside a reference lane, as a developer runs
    it: the common runner with this lane's native drivers, reference interpreter and OpenSSL and, for a staged-samples
    reduction, the lane's stage, writing its run under `artifacts`. Returns its exit status and run directory, which a
    replay refused before validation lacks."""
    import asyncio
    import sys
    root = Path(__file__).resolve().parents[2]
    drivers = [item for option, name in (('--plugin-driver', 'LIVE_TEST_PLUGIN_DRIVER'),
                                         ('--scene-transport-driver', 'LIVE_TEST_SCENE_DRIVER'))
               if name in environment for item in (option, environment[name])]
    if read_json(Path(path).parent / 'reduction.json', limit=16 * 1024 * 1024)['context']['kind'] in STAGED_KINDS:
        drivers += ['--staged-package', environment['LIVE_TEST_STAGED_PACKAGE']]
    process = await asyncio.create_subprocess_exec(
        sys.executable, '-B', str(root / 'tools/live/test.py'), '--reduced-plan', str(path),
        '--artifact-dir', str(artifacts), '--reference-python', sys.executable,
        '--reference-openssl', environment['LIVE_TEST_OPENSSL'], *drivers, '--timeout-ms', '120000', *options,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=str(root), env=dict(environment))
    try:
        # Failure-only watchdog: a healthy replay ends with its single trial.
        output, _ = await asyncio.wait_for(process.communicate(), 300)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / 'command.log').write_bytes(output)
    runs = sorted(artifacts.glob('run-*'))
    return process.returncode, runs[0] if len(runs) == 1 else None


async def check_standalone(test, root):
    """The standalone replay of the self-test reduction in `root`, from `test`, a reference lane's test case: each
    recorded plan reproduces with its named test-only defect (1) and not without it (2), and an altered plan or
    recorded hash is refused (3) before any trial."""
    replays = root.parent / (root.name + '-standalone')
    for name, options, status in (('best.plan.json', (), REPRODUCED), ('original.plan.json', (), REPRODUCED),
                                  ('best.plan.json', ('--without-test-defect',), NOT_REPRODUCED),
                                  ('original.plan.json', ('--without-test-defect',), NOT_REPRODUCED)):
        directory = replays / f"{name.split('.')[0]}{'-without-defect' if options else ''}"
        code, run = await standalone(root / name, directory, *options)
        test.assertEqual(code, status, (directory / 'command.log').read_text(errors='replace')[-8192:])
        replayed = json.loads((run / 'manifest.json').read_text())['results'][0]['reduced_plan']
        target = read_json(root / 'reduction.json')['target']
        test.assertEqual((replayed['plan'], replayed['reproduced'], replayed['target']),
                         (name, status == REPRODUCED, target))
        test.assertEqual(replayed['trial_outcome'], 'mismatch' if status == REPRODUCED else 'passed', replayed)
        test.assertTrue((run / 'reduced-plan-replay').is_dir())
    for alteration in ('plan', 'hash'):
        altered = replays / f'altered-{alteration}' / root.name
        altered.mkdir(parents=True)
        plan = read_json(root / 'best.plan.json')
        summary = read_json(root / 'reduction.json')
        if alteration == 'plan':
            # Without its defect the plan is still a valid seeded selection, which only the recorded hash refuses.
            del plan['test_service_fault']
        else:
            summary['best_sha256'] = '0' * 64
        (altered / 'best.plan.json').write_text(json.dumps(plan))
        (altered / 'reduction.json').write_text(json.dumps(summary))
        code, run = await standalone(altered / 'best.plan.json', altered.parent / 'runs')
        log = (altered.parent / 'runs/command.log').read_text(errors='replace')
        test.assertEqual((code, run), (REFUSED, None), log[-8192:])
        test.assertIn('differs from the hash', log)


def evidence(recipe, result):
    """The lane's record of one replay: what ran, against which target, and its trial result."""
    return dict(schema_version=1, kind=recipe['kind'], plan=recipe['name'], plan_sha256=recipe['plan_sha256'],
                without_test_defect=recipe['without_test_defect'], target=recipe['target'], trial=result,
                reproduced=reproduced(result, recipe['target']))
