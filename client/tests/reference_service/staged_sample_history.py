# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The staged-samples case's seeded plans and their selections, its commit oracle and ledger accounting, and the
diagnostics-sample summary, independent of clients and services.

In a plan every participant authors one seeded edit in every round: each participant sample sets `/Example.value` in
the `sample.live` layer it creates, and each seeded plugin writer, which joins the layer of sample `SAMPLES[i % 2]`,
sets `/Example.value` or defines and then sets its own prim `/W<i>`. Every value is distinct, so a commit names the one
edit it carries. A selection, as failure reduction proposes it, keeps some samples, the writers on their layers and
some rounds, each edit exactly as the seed authored it, so a reduced plan replays inputs the seed reproduces.

A commit of the case carries exactly one planned edit and nothing else: a `set` writes only the `default` of the
prim's `value` attribute, and a `define` creates the root prim and its `value` attribute with exactly the fields the
plugin driver's `usd-prim define` authors. Any other command, field or time sample fails the commit.
"""
from collections import Counter
import copy
import json
import random
import re

import scene_graph_model as graph_model
from scene_graph_wire import Command, GraphWireError, Specifier, Token, decode
from writer_history import WriterMismatch

SCENARIO = 'staged-package.staged-samples.v1'
# The participant samples, in creation order; seeded writer i joins the layer of SAMPLES[i % 2].
SAMPLES = ('native', 'python')
# The named test-only defect a plan may carry, which test_staged_samples implements: the service commits -1 in place
# of the value of writer 0's round-2 edit.
CORRUPT = 'corrupt-w0:2'
FAULTS = (CORRUPT,)
REPLAY_SCOPE = 'exact generated inputs; observed OS/gRPC commit order within a round may differ'

DEFAULT = 'default'
# The specifier a `def` prim carries on the wire.
SPECIFIER_DEF = Specifier(0)
# The diagnostics sample's session-list summary on standard error. The sample writes standard error in text mode,
# so on Windows each line ends in a carriage return before the newline.
LIST_SESSIONS = re.compile(r'^list_sessions result=(\d+) count=(\d+)\r?$', re.MULTILINE)


def listed_sessions(stderr):
    """Every (result, count) session-list summary in the diagnostics sample's standard error bytes."""
    return [(int(result), int(count)) for result, count in LIST_SESSIONS.findall(stderr.decode(errors='replace'))]


def fields(delta):
    """A delta's fields as (section, key, value), without their set order."""
    return sorted(((field.section, field.key, field.value) for field in delta.fields), key=repr)


def planned_edit(delta, planned, attributes):
    """The (name, number, edit) of the planned edit `delta` carries, removed from `planned`, which maps each planned
    value to its edit. `attributes` maps a prim path to the section ID of its `value` attribute in the final scene.
    Raises AssertionError unless the delta is exactly that edit's payload."""
    values = [field.value for field in delta.fields if field.key == DEFAULT]
    if len(values) != 1 or type(values[0]) is not int or values[0] not in planned:
        raise AssertionError(f'carries no single planned edit: {delta}')
    name, number, edit = planned[values[0]]
    attribute = attributes.get(edit['path'])
    if attribute is None:
        raise AssertionError(f"{edit['path']} has no value attribute in the final scene")
    if edit['kind'] == 'set':
        commands, expected = (), [(attribute, DEFAULT, edit['value'])]
    else:
        prim = next((command.section for command in delta.commands if command.spec_type == graph_model.PRIM), None)
        commands = (Command(tag=1, section=prim, parent=1, name=edit['path'][1:], spec_type=graph_model.PRIM),
                    Command(tag=1, section=attribute, parent=prim, name='value', spec_type=graph_model.ATTRIBUTE))
        expected = [(prim, 'specifier', SPECIFIER_DEF), (prim, 'typeName', Token('Scope')),
                    (attribute, 'typeName', Token('int')), (attribute, DEFAULT, edit['value'])]
    actual = (delta.is_diff, delta.base_version, delta.source_format, delta.commands, fields(delta), delta.samples)
    if actual != (False, 0, '', commands, sorted(expected, key=repr), ()):
        raise AssertionError(f"carries more or other than the planned {edit['kind']} of {edit['path']}: {delta}")
    del planned[values[0]]
    return name, number, edit


def generate(master_seed, clients, rounds):
    """The seeded edits of one case, drawn before any participant starts: per round, one edit of /Example.value by each
    participant sample, and one edit by each writer of /Example.value or of its own prim /W<i>, which its first edit of
    it defines."""
    if (type(master_seed) is not int or not 0 <= master_seed < 2**64 or type(clients) is not int or
            not 1 <= clients <= 8 or type(rounds) is not int or not 1 <= rounds <= 32):
        raise ValueError('a staged-samples plan requires a uint64 seed, 1..8 writers and 1..32 rounds')
    rng = random.Random(f'staged-samples:{master_seed}:{clients}:{rounds}')
    values = iter(rng.sample(range(1, 2**31 - 1), (len(SAMPLES) + clients) * rounds))
    samples = {name: [dict(participant=name, ordinal=number, edit=f'{name}:{number}', path='/Example', kind='set',
                           value=next(values)) for number in range(rounds)] for name in SAMPLES}
    writers = []
    for index in range(clients):
        edits, defined = [], False
        for number in range(rounds):
            if rng.random() < 0.5:
                path, kind = '/Example', 'set'
            else:
                path, kind, defined = f'/W{index}', 'set' if defined else 'define', True
            edits.append(dict(participant=f'w{index}', writer=index, ordinal=number, edit=f'w{index}:{number}',
                              path=path, kind=kind, value=next(values)))
        writers.append(edits)
    return dict(schema_version=1, scenario=SCENARIO, master_seed=master_seed, client_count=clients, rounds=rounds,
                samples=samples, writers=writers, layers=[SAMPLES[index % len(SAMPLES)] for index in range(clients)],
                replay_scope=REPLAY_SCOPE)


def selection(plan):
    """The retained samples, seeded writer indices and round ordinals of `plan`."""
    samples = [name for name in SAMPLES if name in plan['samples']]
    return samples, [edits[0]['writer'] for edits in plan['writers']], [op['ordinal'] for op in
                                                                       plan['samples'][samples[0]]]


def select(original, samples, writers, ordinals):
    """The plan of generated plan `original`'s samples `samples`, its writers `writers`, each on the layer of a retained
    sample, in its rounds `ordinals`; `origin` names the population and round count `original` was generated with.
    Raises ValueError for an invalid selection or one in which a writer sets its prim in a retained round without the
    round that defines it."""
    if (not samples or [name for name in SAMPLES if name in samples] != list(samples) or
            sorted(set(writers)) != list(writers) or sorted(set(ordinals)) != list(ordinals) or not ordinals or
            any(not 0 <= w < original['client_count'] or original['layers'][w] not in samples for w in writers) or
            ordinals[0] < 0 or ordinals[-1] >= original['rounds']):
        raise ValueError('invalid sample, writer or round selection')
    plan = copy.deepcopy(original)
    plan.update(client_count=len(writers), rounds=len(ordinals),
                samples={name: [copy.deepcopy(original['samples'][name][o]) for o in ordinals] for name in samples},
                writers=[[copy.deepcopy(original['writers'][w][o]) for o in ordinals] for w in writers],
                layers=[original['layers'][w] for w in writers],
                origin=dict(client_count=original['client_count'], rounds=original['rounds']))
    for edits in plan['writers']:
        defined = set()
        for op in edits:
            if op['kind'] == 'define':
                defined.add(op['path'])
            elif op['path'] != '/Example' and op['path'] not in defined:
                raise ValueError(f"{op['edit']} sets {op['path']}, which no retained round defines")
    return plan


def origin(plan):
    """The generated plan `plan` is, or was selected from; raises ValueError for any other plan or an unknown
    test-only defect."""
    if not isinstance(plan, dict) or plan.get('scenario') != SCENARIO:
        raise ValueError('invalid staged-samples plan')
    if 'test_service_fault' in plan and plan['test_service_fault'] not in FAULTS:
        raise ValueError('unknown test-only service defect')
    try:
        source = plan.get('origin') or dict(client_count=plan['client_count'], rounds=plan['rounds'])
        if not isinstance(source, dict) or set(source) != {'client_count', 'rounds'}:
            raise ValueError('invalid plan origin')
        original = generate(plan['master_seed'], source['client_count'], source['rounds'])
        expected = select(original, *selection(plan)) if 'origin' in plan else copy.deepcopy(original)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('invalid staged-samples plan') from error
    if 'test_service_fault' in plan:
        expected['test_service_fault'] = plan['test_service_fault']
    if json.dumps(plan, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('staged-samples plan differs from its seeded origin selection')
    return original


def validate(plan):
    """A generated plan or a selection of one."""
    origin(plan)
    return plan


def reselect(plan, samples, writers, ordinals):
    """The selection of `plan`'s seeded origin with `samples`, `writers` and `ordinals`, carrying the plan's test-only
    defect."""
    candidate = select(origin(plan), samples, writers, ordinals)
    if 'test_service_fault' in plan:
        candidate['test_service_fault'] = plan['test_service_fault']
    return candidate


def participants(plan, layer=None):
    """Each participant of `plan`, or of the sample layer `layer`, as (name, edits): the samples in creation order, then
    the writers in seeded order."""
    result = [(name, plan['samples'][name]) for name in SAMPLES if name in plan['samples'] and layer in (None, name)]
    return result + [(edits[0]['participant'], edits) for edits, owner in zip(plan['writers'], plan['layers'])
                     if layer in (None, owner)]


def edits(plan):
    """Every edit name of `plan`, in plan order (first_edit)."""
    ranked = [(op['ordinal'], position, op['edit']) for position, (_, ops) in enumerate(participants(plan))
              for op in ops]
    return [name for _, _, name in sorted(ranked)]


def first_edit(plan, names):
    """The first of `names` in plan order: by seeded round, then by participant, the samples in creation order before
    the writers in seeded order. No deletion of samples, writers or rounds renumbers either, so a signature naming this
    edit survives deleting another discrepancy; the commit order within a round is OS-scheduled and never ranks.
    None if `names` is empty."""
    rank = {op['edit']: (op['ordinal'], SAMPLES.index(op['participant']) if 'writer' not in op else
                         len(SAMPLES) + op['writer']) for _, ops in participants(plan) for op in ops}
    return min(names, key=lambda name: (name not in rank, rank.get(name, ()), name), default=None)


def paths(delta, attributes):
    """The prim paths `delta` writes: a root prim it creates, or a prim whose `value` attribute, by section ID in
    `attributes` (prim path to section), it sets."""
    sections = {section: path for path, section in attributes.items()}
    created = {'/' + command.name for command in delta.commands
               if command.spec_type == graph_model.PRIM and command.parent == 1 and command.name}
    return sorted(created | {sections[field.section] for field in delta.fields
                             if field.key == DEFAULT and field.section in sections})


def account(plan, layer, commits, attributes, boundaries, through):
    """The commit accounting of sample layer `layer` after its first `through` retained rounds: every commit after the
    layer's creation carries exactly one planned edit of the layer's participants, each edit commits once and in its
    round, and each participant commits under one wire identity of its own. `commits` are the layer's
    (sequence, wire writer ID, payload) after its creation, `attributes` maps each prim path to the section ID of its
    `value` attribute in the service's scene, and `boundaries` holds the layer's commit count after each earlier round.
    Returns the scene the commits make: every prim path and each prim's last committed value.

    A mismatch raises WriterMismatch. Each class is decided over the whole layer, in a fixed order, so which commit the
    OS scheduled first within a round never selects one: a commit that carries no single planned edit names the layer
    and the least path it writes that the layer's participants author, a repeated, missing or misplaced edit names the
    first such edit in plan order (first_edit), and a wire identity names the first such participant. The commits,
    their sequences and every other discrepancy are evidence.
    """
    members = participants(plan, layer)
    planned = {op['value']: (name, op['ordinal'], op) for name, ops in members for op in ops[:through]}
    authored = {op['path'] for _, ops in members for op in ops}
    position = {op['edit']: index for _, ops in members for index, op in enumerate(ops)}
    carried, unexplained = [], {}
    for sequence, writer, payload in commits:
        try:
            delta = decode(payload)
        except GraphWireError as error:
            unexplained[sequence] = dict(paths=[], error=str(error))
            carried.append(None)
            continue
        try:
            carried.append(planned_edit(delta, dict(planned), attributes)[2])
        except AssertionError as error:
            unexplained[sequence] = dict(paths=paths(delta, attributes), error=str(error))
            carried.append(None)
    if unexplained:
        named = sorted({path for entry in unexplained.values() for path in entry['paths'] if path in authored})
        evidence = dict(commits={str(sequence): entry for sequence, entry in sorted(unexplained.items())})
        if named:
            raise WriterMismatch('commit carries no single planned edit', layer=layer, path=named[0],
                                 evidence=evidence)
        raise WriterMismatch('commit writes no path the layer authors', layer=layer, evidence=evidence)
    names = [op['edit'] for op in carried]
    repeated = {name for name, count in Counter(names).items() if count > 1}
    if repeated:
        raise WriterMismatch('duplicate committed edit', edit=first_edit(plan, repeated),
                             evidence=dict(repeated=sorted(repeated)))
    missing = {op['edit'] for _, _, op in planned.values()} - set(names)
    if missing:
        raise WriterMismatch('admitted edit missing from service ledger', edit=first_edit(plan, missing),
                             evidence=dict(missing=sorted(missing), sequences=[c[0] for c in commits]))
    misplaced = {op['edit']: sum(index >= boundary for boundary in boundaries)
                 for index, op in enumerate(carried) if sum(index >= boundary for boundary in boundaries) !=
                 position[op['edit']]}
    if misplaced:
        raise WriterMismatch('edit committed outside its round', edit=first_edit(plan, misplaced),
                             evidence=dict(rounds=dict(sorted(misplaced.items()))))
    wire = {}
    changed = set()
    for (_, writer, _), op in zip(commits, carried):
        owner = op['participant']
        if not writer or wire.setdefault(owner, writer) != writer:
            changed.add(owner)
    order = [name for name, _ in members]
    if changed:
        raise WriterMismatch('participant identity changed within workload',
                             participant=min(changed, key=order.index), evidence=dict(participants=sorted(changed)))
    shared = [name for name in order if name in wire and list(wire.values()).count(wire[name]) > 1]
    if shared:
        raise WriterMismatch('participants share a wire writer identity', participant=shared[0],
                             evidence=dict(participants=shared))
    scene = {'/Example': 0}
    for op in carried:
        scene[op['path']] = op['value']
    return dict(paths=sorted(scene), values=scene)
