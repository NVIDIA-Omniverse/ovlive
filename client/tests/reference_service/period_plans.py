# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seeded period plans of the long-session plugin cases, their selections and truncation, and the oracles of a
backend rotation and a stream-age expiry; no client imports.

The long-rotations, soak and stream-age cases run periods, a backend rotation or a stream-clock step each, in which
every writer authors one root-field edit; period p is ordinal p - 1 of every writer. Two consecutive periods form a
cycle, cycle c holding periods 2c + 1 and 2c + 2: a rotation goes to backend B in odd periods and back to A in even
ones, whatever the selection, and a stream-age expiry ends every stream in each cycle's second period. A selection
keeps some seeded writers and whole cycles, each edit exactly as the seed authored it, and only its last cycle may
stop after its first period. A plan truncated after period p (`truncate`) executes exactly the inputs of the plan it
is cut from through p, so a failure raised in period p replays from it, and the replay is as long as the failure's
prefix rather than the whole plan.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import random

from writer_history import WriterMismatch

LONG_ROTATIONS = 'public-plugin.long-rotations.v1'
STREAM_AGE = 'public-plugin.stream-age.v1'
# Named test-only defects a plan may carry, which test_plugin implements. In either scenario the service commits -1
# for writer 0's ordinal-4 edit, its owned field of period 5: in a stream-age plan the period after the first expiry,
# in a long-rotations plan the third rotation to backend B. In a stream-age plan the harness commits a peer edit that
# the period's frontier does not count before the expiry of period 6, once every writer applied it, so every read
# resumes past that frontier. In a long-rotations plan the harness forwards rotation 6 to the backend it leaves.
FAULTS = {LONG_ROTATIONS: ('corrupt-w0:4', 'misroute-rotation-6'), STREAM_AGE: ('corrupt-w0:4', 'unannounced-peer-6')}


def generate(master_seed, clients, periods, scenario=LONG_ROTATIONS):
    """One root-field edit per writer per period, over a bounded field set.

    Even ordinals cycle through four owned fields and odd ordinals pick one of three shared fields, so the scene
    stays small however many periods run. Every value is unique, so a committed value names its planned edit.
    """
    if scenario not in FAULTS:
        raise ValueError('unknown period plan scenario')
    if not 2 <= clients <= 8 or not 1 <= periods <= 4096:
        raise ValueError('long rotation profile requires 2..8 clients and 1..4096 rotations')
    seeds = [int.from_bytes(hashlib.sha256(f'long-rotation-v1:{master_seed}:{i}'.encode()).digest()[:8], 'big')
             for i in range(clients)]
    writers = []
    for writer, seed in enumerate(seeds):
        rng = random.Random(seed)
        writers.append([{'writer': writer, 'ordinal': step, 'edit': f'w{writer}:{step}',
                         'field': f'owned_{writer}_{step // 2 % 4}' if step % 2 == 0 else f'shared_{rng.randrange(3)}',
                         'value': (writer + 1) * 10000000 + step * 1024 + rng.randrange(1024)}
                        for step in range(periods)])
    return {'schema_version': 1, 'scenario': scenario, 'master_seed': master_seed,
            'writer_seeds': seeds, 'client_count': clients, 'writers': writers,
            'replay_scope': 'exact generated inputs; observed OS/gRPC commit order may differ'}


def periods(plan):
    """The seeded periods the plan runs, in order."""
    return [op['ordinal'] + 1 for op in plan['writers'][0]]


def last_period(plan):
    """The last seeded period of the generated plan `plan` is, or was selected from."""
    return plan['origin']['periods'] if 'origin' in plan else len(plan['writers'][0])


def select(original, writers, ordinals):
    """The plan of generated plan `original`'s seeded writers `writers` in its ordinals `ordinals`.

    The ordinals are whole cycles, except that the last cycle may hold only its first period. Writers keep their
    seeded edits and take their positions in `writers`; `origin` names the population and period count `original`
    was generated with.
    """
    count = len(original['writers'][0])
    retained = set(ordinals)
    if (not writers or not ordinals or sorted(set(writers)) != writers or sorted(retained) != ordinals or
            writers[0] < 0 or writers[-1] >= original['client_count'] or ordinals[0] < 0 or ordinals[-1] >= count or
            any(ordinal ^ 1 not in retained and not (ordinal % 2 == 0 and ordinal == ordinals[-1])
                for ordinal in ordinals)):
        raise ValueError('invalid writer or period selection')
    plan = copy.deepcopy(original)
    plan.update(client_count=len(writers),
                writers=[[copy.deepcopy(original['writers'][w][o]) for o in ordinals] for w in writers],
                origin=dict(client_count=original['client_count'], periods=count))
    return plan


def origin(plan):
    """The generated plan `plan` is, or was selected from; raises ValueError for any other plan or an unknown
    test-only defect."""
    if not isinstance(plan, dict) or plan.get('scenario') not in FAULTS:
        raise ValueError('invalid period plan')
    if 'test_service_fault' in plan and plan['test_service_fault'] not in FAULTS[plan['scenario']]:
        raise ValueError('unknown test-only service defect')
    try:
        source = plan.get('origin') or dict(client_count=plan['client_count'], periods=len(plan['writers'][0]))
        if not isinstance(source, dict) or set(source) != {'client_count', 'periods'}:
            raise ValueError('invalid plan origin')
        original = generate(plan['master_seed'], source['client_count'], source['periods'], plan['scenario'])
        if 'origin' in plan:
            expected = select(original, [writer[0]['writer'] for writer in plan['writers']],
                              [op['ordinal'] for op in plan['writers'][0]])
        else:
            expected = copy.deepcopy(original)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('invalid period plan') from error
    if 'test_service_fault' in plan:
        expected['test_service_fault'] = plan['test_service_fault']
    if json.dumps(plan, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('period plan differs from its seeded origin selection')
    return original


def validate(plan):
    """A generated plan or a selection of one; any selection executes, since root-field edits depend on none."""
    origin(plan)
    return plan


def reselect(plan, writers, ordinals):
    """The selection of `plan`'s seeded origin with `writers` and `ordinals`, carrying the plan's test-only defect."""
    candidate = select(origin(plan), writers, ordinals)
    if 'test_service_fault' in plan:
        candidate['test_service_fault'] = plan['test_service_fault']
    return candidate


def truncate(plan, period):
    """`plan` cut after its seeded period `period`, or None when it does not run that period or ends with it."""
    ordinals = [op['ordinal'] for op in plan['writers'][0]]
    if type(period) is not int or period - 1 not in ordinals or ordinals[-1] == period - 1:
        return None
    return reselect(plan, [writer[0]['writer'] for writer in plan['writers']],
                    [ordinal for ordinal in ordinals if ordinal < period])


@contextmanager
def in_period(period):
    """Name the seeded `period` in the signature of a WriterMismatch raised inside, unless it names one already.

    The period is the failing step's own ordinal, which no deletion of writers or earlier cycles renumbers, and the
    truncation unit cuts the plan after it.
    """
    try:
        yield
    except WriterMismatch as error:
        if 'period' in error.signature:
            raise
        details = {key: value for key, value in error.signature.items() if key != 'reason'}
        raise WriterMismatch(error.signature['reason'], period=period, evidence=error.evidence,
                             **details) from error


def check_expiry_resume(period, writers, resumed, generations, frontier):
    """A stream-age expiry's own oracle over each writer's first read resume after it: the resume is at a later
    stream generation than `generations` (by plan position), from `frontier`, the ledger's last sequence before the
    expiry.

    `writers` are the seeded writer indices by plan position and `resumed` each one's `stream_resumed` record fields
    from its first read batch holding one. Only each writer's first resume is classified: it is the first record of
    its kind after the expiry, whatever the batching, while whether a second resume shares its batch or arrives in a
    later read depends on scheduling. Classes are decided over every writer, in this order, so which writer the
    harness read first never selects one; a failure raises WriterMismatch whose signature is its reason and the
    seeded `period`, and whose evidence is the writers and records, which depend on the rest of the plan. A writer
    with no resume, or with a second one in that batch, raises a plain AssertionError, as the case's later check
    that no read resumed outside an expiry does for a second resume read later, so no reduction trial matches it.
    """
    def fail(reason, failing, **evidence):
        raise WriterMismatch(reason, period=period, evidence=dict(writers=failing, **evidence))
    if any(not records for records in resumed):
        raise AssertionError(dict(reason='read did not resume after a stream-age expiry', period=period,
                                  writers=[w for w, records in zip(writers, resumed) if not records]))
    stalled = [writer for writer, records, before in zip(writers, resumed, generations)
               if records[0]['stream_generation'] <= before]
    if stalled:
        fail('read stream generation did not advance across a stream-age expiry', stalled,
             generations=list(generations), resumed=[records[0]['stream_generation'] for records in resumed])
    moved = [writer for writer, records in zip(writers, resumed) if records[0]['resume_sequence'] != frontier]
    if moved:
        fail('read did not resume at its coverage frontier after a stream-age expiry', moved, frontier=frontier,
             resume_sequences=[records[0]['resume_sequence'] for records in resumed])
    repeated = [writer for writer, records in zip(writers, resumed) if len(records) > 1]
    if repeated:
        raise AssertionError(dict(reason='read resumed more than once after a stream-age expiry', period=period,
                                  writers=repeated, resumed=resumed))


def check_rotation_reconnects(records, generation, initial_ms, maximum_ms):
    """One writer's read reconnects across one backend rotation, over its read records since the previous rotation:
    the read retries from its stream generation `generation` in the first backoff window and resumes exactly once,
    at a later generation. Returns the `retry_scheduled` and `stream_resumed` record fields.

    The first retry's wait, below twice `initial_ms`, shows that the reconnect attempt count reset at the previous
    rotation's recovery readiness; an unreset count would widen the window every rotation and fail the layer after
    reconnect_max_attempts. A reconnect attempt may itself fail before it resumes: the first wait can be near zero
    (design section 12), so the retry can reach gRPC before the channel has seen the cut connection close and fail
    UNAVAILABLE at once, without reaching a backend (#413). Such a retry is the next consecutive attempt, so retry
    k, counted from 1, waits in attempt k's window, below 2 * min(`maximum_ms`, `initial_ms` * 2^(k-1)), from a later
    generation than the retry before it. The layer's own reconnect_max_attempts bounds the retries, and the case
    counts the reads that reach the service. A failure raises a plain AssertionError: retry accounting is not a
    reducible writer mismatch.
    """
    retries = [r['fields'] for r in records if r['event_name'] == 'retry_scheduled']
    resumed = [r['fields'] for r in records if r['event_name'] == 'stream_resumed']
    if len(resumed) != 1 or not retries:
        raise AssertionError(dict(reason='read did not reconnect once across a rotation', retries=retries,
                                  resumed=resumed))
    if (retries[0]['attempt_id'], retries[0]['stream_generation']) != (1, generation):
        raise AssertionError(dict(reason='first read retry is not from the stream the rotation ended',
                                  generation=generation, retries=retries))
    previous = generation - 1
    for attempt, retry in enumerate(retries, 1):
        window = 2 * min(maximum_ms, initial_ms * 2 ** (attempt - 1))
        if retry['stream_generation'] <= previous or retry['delay_ms'] >= window:
            raise AssertionError(dict(reason='read retry outside its consecutive reconnect attempt window',
                                      attempt=attempt, window_ms=window, retries=retries))
        previous = retry['stream_generation']
    if resumed[0]['stream_generation'] <= previous:
        raise AssertionError(dict(reason='read resumed at a stream generation it already retried from',
                                  retries=retries, resumed=resumed))
    return retries, resumed


def check_rotation_resume(period, reads, writers, backend, cursor):
    """A backend rotation's own oracle, once every writer is ready on the rotation's destination: `writers` distinct
    writers, by the client ID of `reads`, the service's read records since the rotation began, resumed on `backend` at
    `cursor`, the frontier before the rotation.

    Each writer's resumed read reached the service before the writer reported ready, so the records are final
    whatever the scheduling, and their order never matters. Writers are counted, not records, so a second reconnect
    of one writer, which scheduling may or may not add, never stands in for another writer's resume. A failure raises
    WriterMismatch whose signature is the reason and the seeded `period`; the backend, cursor and reads are evidence,
    so deleting another writer or cycle keeps the signature. Extra reads are left to the case's plain checks.
    """
    matching = len({read['writer'] for read in reads if read['backend'] == backend and read['cursor'] == cursor})
    if matching < writers:
        raise WriterMismatch('writers did not resume reading on the rotation destination at its frontier',
                             period=period, evidence=dict(backend=backend, cursor=cursor, matching=matching,
                                                          writers=writers, reads=[dict(read) for read in reads]))
