# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selections of a seeded round-based writer plan, as failure reduction proposes them, and the signatures of
its stalled rounds and failed rotations; no client imports.

In a structural, sample or root-field rotation plan (rotation_history) every writer authors one edit in every
round. A selection keeps some of the generated writers, rounds and backend rotations, each edit exactly as the
generator authored it, so a reduced plan replays inputs the seed reproduces, never edited values.
"""
from collections import Counter
import copy
import json

from writer_history import WriterMismatch


def select(original, writers, ordinals, rotations):
    """The plan of `original`'s writers `writers` in its rounds `ordinals`, rotating after its rounds `rotations`.

    `original` is a generated plan, whose positions are its ordinals; a selection rotates only after a retained
    round the original rotates after. Writers keep their seeded edits and edit names and take their positions in
    `writers`; each round's release order keeps the original order of the retained writers' slots; `origin` names
    the population and round count `original` was generated with.
    """
    positions = {writer: position for position, writer in enumerate(writers)}
    if (len(positions) != len(writers) or sorted(writers) != writers or sorted(set(ordinals)) != ordinals or
            not set(rotations) <= set(ordinals) & set(original['rotation_after']) or
            sorted(set(rotations)) != rotations or
            not writers or not ordinals or writers[0] < 0 or writers[-1] >= original['client_count'] or
            ordinals[0] < 0 or ordinals[-1] >= original['rounds']):
        raise ValueError('invalid writer, round or rotation selection')
    plan = copy.deepcopy(original)
    plan.update(client_count=len(writers), rounds=len(ordinals),
                writers=[[copy.deepcopy(original['writers'][w][o]) for o in ordinals] for w in writers],
                release_orders=[[positions[w] for w in original['release_orders'][o] if w in positions]
                                for o in ordinals],
                rotation_after=[ordinals.index(o) for o in rotations],
                origin=dict(client_count=original['client_count'], rounds=original['rounds']))
    return plan


def validate_selection(plan, generate, faults):
    """A round-based plan is a generated plan, or `select` of one (it names its `origin`), with an optional named
    test-only service defect from `faults`; returns the generated origin."""
    if not isinstance(plan, dict):
        raise ValueError('invalid round-based writer plan')
    if 'test_service_fault' in plan and plan['test_service_fault'] not in faults:
        raise ValueError('unknown test-only service defect')
    try:
        origin = plan.get('origin') or dict(client_count=plan['client_count'], rounds=plan['rounds'])
        if not isinstance(origin, dict) or set(origin) != {'client_count', 'rounds'}:
            raise ValueError('invalid plan origin')
        original = generate(plan['master_seed'], origin['client_count'], origin['rounds'])
        if 'origin' in plan:
            writers = [writer[0]['writer'] for writer in plan['writers']]
            ordinals = [op['ordinal'] for op in plan['writers'][0]]
            expected = select(original, writers, ordinals, [ordinals[p] for p in plan['rotation_after']])
        else:
            expected = copy.deepcopy(original)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('invalid round-based writer plan') from error
    if 'test_service_fault' in plan:
        expected['test_service_fault'] = plan['test_service_fault']
    if json.dumps(plan, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('writer plan differs from its seeded origin selection')
    return original


def first_difference(actual, expected):
    """The first scene path, in path order, whose projected node differs between two scenes, or is in only one."""
    def by_path(nodes):
        return {node['path']: json.dumps(node, sort_keys=True, allow_nan=False) for node in nodes}
    left, right = by_path(actual), by_path(expected)
    return next((path for path in sorted(set(left) | set(right)) if left.get(path) != right.get(path)), None)


def check_replicas(plan, reports, expected, attribute):
    """The client-versus-service scene comparison of a round-based case: every one of `reports`, native writer
    reports at one convergence point, holds an authoritative and a visible scene equal to `expected`, the scene the
    service ledger replays to. Each scene carries sibling ranks in place of order values (scene_graph_model.ranked).

    A difference raises WriterMismatch whose signature is the reason, the authoritative scenes before the visible
    ones, and the first in plan order (first_edit) of the edits `attribute(actual, expected, path)` names for each
    differing report's first differing path. `attribute` is the class's scene attribution over its oracle's lineage,
    which reads only the identities, sections and sample keys at that path, never values, the report's position or
    the commit order, so neither reader order nor the values a defect delivers change the signature. Which reports
    differ, their paths and both nodes at each are evidence.
    """
    for view in ('authoritative', 'visible'):
        differing = [(position, first_difference(report[view], expected)) for position, report in enumerate(reports)]
        differing = [(position, path) for position, path in differing if path is not None]
        if not differing:
            continue

        def node(scene, path):
            return next((item for item in scene if item['path'] == path), None)
        edits = {attribute(reports[position][view], expected, path) for position, path in differing}
        raise WriterMismatch(f'writer {view} scene differs from the service ledger',
                             edit=first_edit(plan, edits - {None}),
                             evidence=dict(readers=[dict(position=position, path=path,
                                                         actual=node(reports[position][view], path),
                                                         expected=node(expected, path))
                                                    for position, path in differing]))


def first_edit(plan, edits, peers=None):
    """The first of `edits` in commit order, by round position and then writer position; `peers` maps each peer
    effect's name to the round position it follows, after that round's writers, as in `latest_edit`. An edit
    neither holds follows every ranked one, in name order. A structural or sample signature names this edit, so
    deleting a later discrepancy leaves it unchanged."""
    rank = {op['edit']: (index, False, position)
            for position, writer in enumerate(plan['writers']) for index, op in enumerate(writer)}
    rank.update({name: (index, True, 0) for name, index in (peers or {}).items()})
    return min(edits, key=lambda name: (name not in rank, rank.get(name, ()), name), default=None)


def admission_discrepancy(plan, planned, admissions):
    """The first discrepant edit of a missing, repeated or unplanned admission, and every such edit as evidence."""
    names = [admission['edit'] for admission in admissions]
    missing, unplanned = set(planned) - set(names), set(names) - set(planned)
    repeated = {name for name, count in Counter(names).items() if count > 1}
    return first_edit(plan, missing | repeated | unplanned), dict(
        missing=sorted(missing), repeated=sorted(repeated), unplanned=sorted(unplanned))


def latest_edit(entries):
    """The name of the latest of `entries` (lineage entries with `round` and `writer` positions) in plan order.

    Rounds are barriers, so a later round is later; within a round a peer effect (writer None) follows the writers,
    and among writers the first in plan position is named, since their commit order is OS-scheduled. None if empty.
    """
    def rank(name):
        entry = entries[name]
        return entry['round'], entry['writer'] is None, -(entry['writer'] or 0)
    return max(entries, key=rank, default=None)


def stopped_short(account, *, frontier, sequence, in_flight, refused, stopped):
    """Classify a round whose ledger stops at `sequence`, short of its `frontier`, the sequence every writer then
    waits to apply; without this the writers' convergence wait times out.

    The ledger is final, and the failure semantic, only when no UpdateLayer call is `in_flight`, no write was
    `refused` retryably during the round, so no writer re-sends, and no native writer has `stopped` (positions of
    exited or failed driver processes). Otherwise the stall is a watchdog or crash matter and raises a plain
    AssertionError, which no reduction trial can match. A final short ledger lacks an admitted edit, so `account()`,
    the class's ledger accounting through the round, raises its WriterMismatch; the signature is this reason, that
    mismatch's reason as `cause` and its other signature fields, such as the first lacking edit in plan order. The
    frontier, sequence and the accounting evidence depend on the rest of the plan and are evidence.
    """
    if in_flight or refused or stopped:
        raise AssertionError(dict(reason='round commits stopped short of its frontier without a final ledger',
                                  frontier=frontier, sequence=sequence, in_flight=in_flight, refused=refused,
                                  stopped=stopped))
    try:
        account()
    except WriterMismatch as error:
        details = {key: value for key, value in error.signature.items() if key != 'reason'}
        raise WriterMismatch('round commits stopped short of its frontier', cause=error.signature['reason'],
                             evidence=dict(frontier=frontier, sequence=sequence, accounting=error.evidence),
                             **details) from error
    raise AssertionError('round commits stopped short of its frontier, yet the ledger accounts')


def check_rotation(ordinal, generations, resumed, reads, backend, cursor):
    """A backend rotation's own oracle, after its writers converged on the destination: every writer's read
    `generation` (by plan position, before the rotation) is below its `resumed` one, and every writer, by the client
    ID of `reads`, every read record so far, resumed on `backend` at `cursor`, the frontier before it.

    A failure raises WriterMismatch whose signature is the reason and `rotation`, `ordinal` of the round the
    rotation follows in the seeded plan; the writers, generations, backend and cursor are evidence, so deleting
    another writer or round keeps the signature.
    """
    stalled = [position for position, (old, new) in enumerate(zip(generations, resumed)) if new <= old]
    if stalled:
        raise WriterMismatch('rotation did not advance a writer read generation', rotation=ordinal,
                             evidence=dict(writers=stalled, generations=list(generations), resumed=list(resumed)))
    check_resume(ordinal, reads, len(generations), backend, cursor)


def check_resume(ordinal, reads, writers, backend, cursor):
    """At least `writers` distinct writers, by the client ID of `reads`, the read records of a rotation, resumed on
    `backend` at `cursor`, the frontier before the rotation.

    Writers are counted, not records, so a second reconnect of one writer, which scheduling may or may not add, never
    stands in for another writer's resume, and the order of `reads` never matters. A failure raises WriterMismatch
    whose signature is the reason and `rotation`, `ordinal` of the round the rotation follows in the seeded plan, and
    whose evidence is the rest. Extra reads are left to the case's plain checks.
    """
    matching = len({read['writer'] for read in reads if read['backend'] == backend and read['cursor'] == cursor})
    if matching < writers:
        raise WriterMismatch('writers did not resume reading on the rotation destination at its frontier',
                             rotation=ordinal, evidence=dict(backend=backend, cursor=cursor, matching=matching,
                                                             writers=writers))
