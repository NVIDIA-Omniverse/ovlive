# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seeded root-field plans of the plugin rotation cases, as round-based plans, and the oracle of their write cut;
no client imports.

A queued-rotation plan rotates the backend between phases of rounds: the first phase is authored connected, and
each later one is queued during the rotation before it. An uncertain-write plan cuts every writer's first write at
one fault boundary and queues the rest behind it. Every writer authors one root-field edit (writer_history.generate)
in every round, so a selection (round_plans.select) keeps some writers, rounds and rotations, each edit exactly as
the seed authored it.
"""
from collections import Counter

import round_plans
import writer_history
from writer_history import WriterMismatch

QUEUED = 'public-plugin.queued-rotations.v2'
UNCERTAIN = 'public-plugin.uncertain-rotation.v2'
# Named test-only defects a plan may carry, which test_plugin implements: the harness forwards the rotation after
# round 7 to the backend it leaves; CorruptRotationWriterService commits -1 for writer 0's round-6 edit, an owned
# field; CutWriterService commits writer 0's round-0 write when the cut cancels its handler held before execution,
# or passes that write through its commit step without committing it; and the harness releases writer 0's round-0
# write, held after its commit, before the cut while writer 0 has queued writes, which then follow it.
FAULTS = {QUEUED: ('misroute-rotation-7', 'corrupt-owned-w0:6'),
          UNCERTAIN: ('apply-cut-w0:0', 'lose-cut-w0:0', 'leak-cut-w0:0')}
PHASE_ROUNDS = 4


def _generate(scenario, seed, clients, rounds, rotation_after):
    base = writer_history.generate(seed, clients, rounds)
    return dict(schema_version=1, scenario=scenario, master_seed=seed, writer_seeds=base['writer_seeds'],
                client_count=clients, rounds=rounds, writers=base['writers'],
                release_orders=[list(base['release_order']) for _ in range(rounds)], rotation_after=rotation_after,
                replay_scope='generated inputs and named gates; observed commit order may differ')


def generate_queued(seed, clients=4, rounds=12):
    """Phases of PHASE_ROUNDS rounds, with a backend rotation after each phase but the last."""
    if not 2 <= clients <= 8 or rounds % PHASE_ROUNDS or not PHASE_ROUNDS <= rounds <= 16:
        raise ValueError('queued-rotation profile requires 2..8 writers and 4, 8, 12 or 16 rounds')
    return _generate(QUEUED, seed, clients, rounds, list(range(PHASE_ROUNDS - 1, rounds - 1, PHASE_ROUNDS)))


def generate_uncertain(seed, clients=4, rounds=4):
    """One cut, which is the scenario and no rotation unit; each writer's first retained edit is the cut write."""
    if not 2 <= clients <= 8 or not 2 <= rounds <= 8:
        raise ValueError('uncertain-write profile requires 2..8 writers and 2..8 rounds')
    return _generate(UNCERTAIN, seed, clients, rounds, [])


def origin(plan):
    """The generated plan `plan` is, or was selected from (round_plans.select)."""
    if not isinstance(plan, dict) or plan.get('scenario') not in FAULTS:
        raise ValueError('invalid root-field rotation plan')
    generate = generate_queued if plan['scenario'] == QUEUED else generate_uncertain
    return round_plans.validate_selection(plan, generate, FAULTS[plan['scenario']])


def validate(plan):
    """A generated plan or a selection of one; any selection executes, since root-field edits depend on none."""
    origin(plan)
    return plan


def phases(plan):
    """The [first, end) round positions of each phase: the rounds before the first rotation, then those after each
    rotation. A rotation after the last retained round leaves its phase empty, so the writers only resume."""
    bounds = [0] + [position + 1 for position in plan['rotation_after']] + [plan['rounds']]
    return list(zip(bounds, bounds[1:]))


def rotation_ordinal(plan, index):
    """The seeded ordinal of the round the plan's `index`th rotation follows, which names the rotation."""
    return plan['writers'][0][plan['rotation_after'][index]]['ordinal']


def check_cut(plan, boundary, commits):
    """The cut's own ledger oracle over `commits`, the decoded writer commits after the harness peer writes.

    A writer's first retained edit is its cut write and the rest are queued behind it. A write cut `before`
    execution never commits, one cut `after_commit` is in the ledger exactly once, and a queued write never reaches
    the service. A failure raises WriterMismatch whose signature is its reason and the first discrepant edit in plan
    order (round, then writer), so the commit order within the cut and the other discrepancies are evidence. An
    unexplained commit is named by its field names, which the plan authors; among several, by the least sorted
    names, so their commit order is evidence too.
    """
    names = {(op['field'], op['value']): op['edit'] for writer in plan['writers'] for op in writer}
    cut = [writer[0]['edit'] for writer in plan['writers']]
    unexplained = [commit['fields'] for commit in commits
                   if len(commit['fields']) != 1 or None in {names.get(item) for item in commit['fields'].items()}]
    if unexplained:
        named = min(unexplained, key=sorted)
        raise WriterMismatch('unexplained commit across a write cut', fields=sorted(named),
                             evidence=dict(commit=named, unexplained=unexplained))
    seen = Counter(names[item] for commit in commits for item in commit['fields'].items())
    queued = set(seen) - set(cut)
    if queued:
        raise WriterMismatch('queued write behind a cut write reached the service',
                             edit=round_plans.first_edit(plan, queued), evidence=dict(queued=sorted(queued)))
    repeated = {edit for edit, count in seen.items() if count > 1}
    if boundary == 'before' and seen:
        raise WriterMismatch('write cut before execution committed', edit=round_plans.first_edit(plan, set(seen)),
                             evidence=dict(committed=sorted(seen), repeated=sorted(repeated)))
    if boundary == 'after_commit':
        missing = set(cut) - set(seen)
        if missing:
            raise WriterMismatch('write cut after its commit missing from the ledger',
                                 edit=round_plans.first_edit(plan, missing), evidence=dict(missing=sorted(missing)))
        if repeated:
            raise WriterMismatch('write cut after its commit committed more than once',
                                 edit=round_plans.first_edit(plan, repeated), evidence=dict(repeated=sorted(repeated)))
