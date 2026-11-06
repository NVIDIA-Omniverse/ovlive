# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent public-authoring workload/accounting; no wire, USD or client imports.

The seed reproduces authored inputs and named release choices, not OS scheduling.
The semantic oracle uses explicit public-constructor templates and operation
intent. Native IDs are bound as opaque witnesses, never as expected contents.
"""
import copy
import hashlib
import json
import random
import struct

import round_plans
from writer_history import WriterMismatch

# Named test-only service defects a structural plan may carry; test_graph_transport implements them with
# CorruptGraphWriterService. Each names an edit of the seed-42 plan (SELFTEST_SEED): corrupt-shared-w0:2 commits
# writer 0's round-2 shared-field write as -1, corrupt-reorder-w1:2 commits writer 1's round-2 reorder with its
# children listed in reverse, and relay-shared relays every shared-field write to its readers as -1 while the ledger
# keeps the uploaded bytes.
FAULTS = ('corrupt-shared-w0:2', 'corrupt-reorder-w1:2', 'relay-shared')
# The edit kind each edit-naming fault needs; origin refuses a plan whose generated origin lacks the named edit or
# holds it as another kind, which the defect would misread or never reach.
FAULT_KINDS = {'corrupt-shared-w0:2': 'shared', 'corrupt-reorder-w1:2': 'reorder'}
# The seed of the structural reduction self-tests, whose expected minimal plans hold the edits FAULTS name.
SELFTEST_SEED = 42
# The plan's scenario; validate refuses a plan of any other, so a retained plan of another format is never misread.
SCENARIO = 'private-attachment.structural-v2'
ROUNDS = 12
# The parents a writer's branch lives under: /P and /Q, each with a permanent sentinel child.
PARENTS = (2, 5)
# The parent under which a branch is renamed or its siblings reordered: /P, whose children are all prims. /Q also
# holds the property `other`, and a service that stores a Reorder verbatim interleaves a parent's prims and
# properties in the listed order, while every reader keeps each child list's own positions (design sections 9, 20).
ORDERED_PARENT = 2
SHAPES = ('usd-create', 'usd-variants', 'usd-targets')
# Every generated edit class. `shared` sets a field of /P and needs no branch; every other kind but `create` edits
# the writer's live branch.
KINDS = ('create', 'field', 'shared', 'move', 'rename', 'reorder', 'erase')
NEEDS_BRANCH = ('field', 'move', 'rename', 'reorder', 'erase')
# The value families a field or shared-field edit writes: a scalar int and float, a vector, a string, a token array
# and a float array, each in the JSON projection the native graph driver reports.
VALUE_TYPES = ('int32', 'float', 'vec3d', 'string', 'token[]', 'float[]')
# The relative weight of each kind a writer draws while it holds a branch, and while it holds none.
BRANCH_WEIGHTS = dict(field=2, shared=2, move=2, rename=3, reorder=3, erase=1)
FREE_WEIGHTS = dict(create=3, shared=1)
# The field a reorder edit sets on the parent it reorders, so that every reorder's bytes are distinct.
REORDER_FIELD = 'reordered'


def value(kind, raw):
    return {'type': kind, 'value': raw}


def tokens(*items):
    return value('tokens', list(items))


def path_op(items):
    return value('path_op', dict(explicit=True, explicit_items=list(items), added=[],
                                prepended=[], appended=[], deleted=[], ordered=[]))


def template(shape, prefix, amount):
    """Complete authored fields and graph-parent suffixes, not schema defaults."""
    prim = {'specifier': value('specifier', 0), 'typeName': value('token', 'Scope')}
    child = {'specifier': value('specifier', 0), 'typeName': value('token', '__AnyType__')}
    attr = {'typeName': value('token', 'int'), 'default': value('int32', amount)}
    rel = {'custom': value('bool', True), 'variability': value('variability', 1)}
    if shape == 'usd-create':
        return {'': (6, None, dict(prim, primChildren=tokens('Child'), properties=tokens('link'))),
                '/Child': (6, '', dict(child, properties=tokens('amount'))),
                '/Child.amount': (1, '/Child', attr), '.link': (8, '', rel)}
    if shape == 'usd-variants':
        return {'': (6, None, dict(prim, variantSetChildren=tokens('look'))),
            '{look=}': (11, '', {'variantChildren': tokens('1', 'high-low')}),
            '{look=1}': (10, '{look=}', {}),
            '{look=high-low}': (10, '{look=}', {'primChildren': tokens('Child'), 'properties': tokens('amount'),
                                              'variantSetChildren': tokens('detail')}),
            '{look=high-low}Child': (6, '{look=high-low}', child),
            '{look=high-low}.amount': (1, '{look=high-low}', attr),
            '{look=high-low}{detail=}': (11, '{look=high-low}', {'variantChildren': tokens('2')}),
            '{look=high-low}{detail=2}': (10, '{look=high-low}{detail=}', {'properties': tokens('link')}),
            '{look=high-low}{detail=2}.link': (8, '{look=high-low}{detail=2}', rel)}
    if shape == 'usd-targets':
        paths, targets = (prefix + '/Input.x', '/External.y'), (prefix + '/Target', '/Elsewhere')
        result = {'': (6, None, dict(prim, properties=tokens('amount', 'link'))),
            '.amount': (1, '', dict(attr, connectionChildren=value('paths', [paths[1], paths[0]]), connectionPaths=path_op(paths))),
            '.link': (8, '', dict(rel, targetChildren=value('paths', [targets[1], targets[0]]), targetPaths=path_op(targets)))}
        result.update({'.amount[' + p + ']': (2, '.amount', {}) for p in paths})
        result.update({'.link[' + p + ']': (9, '.link', {}) for p in targets})
        return result
    raise AssertionError('unknown constructor template')


def _bits(form, number):
    """The IEEE-754 bits of `number` in struct format `form` ('f' or 'd'), as the driver reports a component."""
    return struct.unpack('<' + {'f': 'I', 'd': 'Q'}[form], struct.pack('<' + form, number))[0]


def typed(kind, number, writer, ordinal):
    """The `kind` value (VALUE_TYPES) a field edit writes, distinct for each distinct `number`."""
    if kind == 'int32':
        return value(kind, number)
    if kind == 'float':
        return value(kind, [_bits('f', float(number))])
    if kind == 'vec3d':
        # Writer 0's last component is -0.0, whose representation differs from 0.0.
        return value(kind, [_bits('d', component) for component in (float(number), ordinal + 0.5, -float(writer))])
    if kind == 'string':
        return value(kind, f'w{writer}-\u03bb-{number}')
    if kind == 'token[]':
        return value(kind, [f'tok{number}', f'round{ordinal}'])
    if kind == 'float[]':
        return value(kind, [_bits('f', component) for component in (float(number), 0.25, -0.0)])
    raise AssertionError('unknown structural value type')


def _draw(rngs, clients):
    """One draw of every writer's edits, each from its own seeded generator, round by round.

    A writer that holds no branch creates one under /P or /Q or writes the shared field; one that holds a branch
    sets its leaf field or the shared field, moves the branch to the other parent, erases it or, under /P
    (ORDERED_PARENT), renames it or reverses /P's prim children (a same-parent reorder). A rename or reorder
    compiles a Reorder that lists every current child of the parent, which the service stores verbatim, so in its
    round no other edit changes that parent's children: the Reorder then lists exactly the children the service
    holds when it commits, never an omitted or unknown one. Every move changes parent, so it appends and compiles no Reorder. Every created, moved
    or renamed name is unique, so no edit collides with another, and the sentinel children /P/Child and
    /Q/Sentinel remain, so a reorder's parent always holds at least two prim children and every reorder changes
    the order.
    """
    writers = [[] for _ in range(clients)]
    branches = [None] * clients  # Each writer's branch parent, or None.
    counts = [0] * clients  # Each writer's field edits so far.
    orders = [rng.sample(VALUE_TYPES, len(VALUE_TYPES)) for rng in rngs]
    for ordinal in range(ROUNDS):
        touched, locked = set(), set()  # Parents whose children this round's edits change, or reorder.
        for writer, rng in enumerate(rngs):
            parent = branches[writer]
            if parent is None:
                weights = dict(FREE_WEIGHTS)
                if all(p in locked for p in PARENTS):
                    del weights['create']
            else:
                other = PARENTS[1 - PARENTS.index(parent)]
                weights = dict(BRANCH_WEIGHTS)
                if parent in locked:
                    for kind in ('move', 'erase', 'rename', 'reorder'):
                        weights.pop(kind)
                if other in locked:
                    weights.pop('move', None)
                if parent in touched or parent != ORDERED_PARENT:
                    weights.pop('rename', None)
                    weights.pop('reorder', None)
            kinds = sorted(weights)
            kind = rng.choices(kinds, [weights[k] for k in kinds])[0]
            number = (writer + 1) * 100000 + ordinal * 100 + rng.randrange(100)
            op = dict(writer=writer, ordinal=ordinal, edit=f'w{writer}:{ordinal}', kind=kind)
            if kind == 'create':
                parent = rng.choice([p for p in PARENTS if p not in locked])
                op.update(parent=parent, name=f'Writer{writer}_{ordinal}_{rng.randrange(10000)}',
                          shape=rng.choice(SHAPES), value=number)
                touched.add(parent)
            elif kind in ('field', 'shared'):
                op['value'] = typed(orders[writer][counts[writer] % len(VALUE_TYPES)], number, writer, ordinal)
                counts[writer] += 1
            elif kind == 'move':
                touched |= {parent, other}
                parent = other
                op.update(parent=parent, name=f'Moved{writer}_{ordinal}_{rng.randrange(10000)}')
            elif kind == 'rename':
                locked.add(parent)
                op.update(parent=parent, name=f'Renamed{writer}_{ordinal}_{rng.randrange(10000)}')
            elif kind == 'reorder':
                locked.add(parent)
                op.update(parent=parent, value=number)
            else:
                touched.add(parent)
                op['parent'] = parent
                parent = None
            branches[writer] = parent
            writers[writer].append(op)
    return writers


def covers(writers):
    """Whether a draw holds every edit class and value type, and a field two edits write, so a generated plan
    covers deletion and supersession."""
    ops = [op for writer in writers for op in writer]
    return ({op['kind'] for op in ops} == set(KINDS) and sum(op['kind'] == 'shared' for op in ops) >= 2 and
            {op['value']['type'] for op in ops if op['kind'] in ('field', 'shared')} == set(VALUE_TYPES))


def generate(seed, clients=4):
    """The seeded structural plan of `clients` writers: each writer's edits come from its own seed, drawn again
    from the same generators until the plan covers every class (`covers`), so the plan is a pure function of the
    seed."""
    if not 2 <= clients <= 8:
        raise ValueError('bounded structural profile needs 2..8 writers')
    seeds = [int.from_bytes(hashlib.sha256(f'structural-writer-v2:{seed}:{i}'.encode()).digest()[:8], 'big')
             for i in range(clients)]
    rngs = [random.Random(writer_seed) for writer_seed in seeds]
    for _ in range(1000):
        writers = _draw(rngs, clients)
        if covers(writers):
            break
    else:
        raise AssertionError('no seeded structural draw covers every edit class')
    rng = random.Random(seed)
    releases = [rng.sample(range(clients), clients) for _ in range(ROUNDS)]
    return dict(schema_version=2, scenario=SCENARIO, master_seed=seed,
                writer_seeds=seeds, client_count=clients, rounds=ROUNDS, writers=writers,
                release_orders=releases, rotation_after=[5, 9],
                replay_scope='exact generated operations and named gates; observed OS/gRPC order may differ')


def _generated(seed, clients, rounds):
    plan = generate(seed, clients)
    if rounds != plan['rounds']:
        raise ValueError('structural plans have a fixed round count')
    return plan


def origin(plan):
    """The generated plan `plan` is, or was selected from (round_plans.select)."""
    if not isinstance(plan, dict) or plan.get('scenario') != SCENARIO:
        raise ValueError('not a structural-v2 writer plan')
    original = round_plans.validate_selection(plan, _generated, FAULTS)
    fault = plan.get('test_service_fault')
    if fault in FAULT_KINDS:
        edit = fault.rsplit('-', 1)[1]
        kinds = [op['kind'] for writer in original['writers'] for op in writer if op['edit'] == edit]
        if kinds != [FAULT_KINDS[fault]]:
            raise ValueError(f'test-only service defect {fault} names no {FAULT_KINDS[fault]} edit of this plan')
    return original


def validate(plan):
    """A generated plan or a selection of one in which every edit finds the branch state it was generated against.

    A create needs no live branch and a shared-field edit none at all; every other edit needs one. A move changes
    the branch's parent, an erase ends the branch, and an erase, rename or reorder names the parent its branch is
    under, so every edit that changes or reorders a parent's children does so at the parent it was generated for;
    the round's other edits then leave a rename's or reorder's parent alone, as in the generated plan. A selection
    that breaks this, such as a rename without the round that creates its branch or a reorder without the move that
    took the branch to its parent, cannot execute as generated, so reduction never proposes it.
    """
    origin(plan)
    for writer in plan['writers']:
        branch = None
        for op in writer:
            if op['kind'] == 'create':
                if branch is not None:
                    raise ValueError('a writer creates a branch before erasing its previous one')
                branch = op['parent']
            elif op['kind'] == 'shared':
                continue
            elif branch is None:
                raise ValueError('a writer edits a branch it has not created')
            elif op['kind'] == 'move':
                if op['parent'] == branch:
                    raise ValueError('a writer move must change parent')
                branch = op['parent']
            elif op['kind'] in ('erase', 'rename', 'reorder') and op['parent'] != branch:
                raise ValueError('a writer edit names a parent its branch is not under')
            elif op['kind'] == 'erase':
                branch = None
    return plan


def leaf(branch):
    return branch['ids'][{'usd-create': '/Child.amount', 'usd-variants': '{look=high-low}.amount',
                          'usd-targets': '.amount'}[branch['shape']]]


def project(nodes):
    keys = ('id', 'parent', 'type', 'path', 'fields')
    return [{key: copy.deepcopy(node[key]) for key in keys} for node in sorted(nodes, key=lambda n: n['id'])]


def verify_scene(actual, expected, lineage=None):
    """Raise a mismatch naming the edit that last affected the first differing node.

    `lineage` is account()'s lineage of `expected`. Among the committed edits that changed that node, on either
    side, as one of their identities or as the parent whose children they created, moved or erased, the signature
    names the latest in plan order (round_plans.latest_edit); the service commit order within a round is
    OS-scheduled, so it never ranks them. Without a lineage, or for a node no native edit changed, it names none.
    """
    # JSON's typed scalar spelling avoids Python's True == 1 equivalence.
    if json.dumps(project(actual), sort_keys=True) != json.dumps(expected, sort_keys=True):
        path = round_plans.first_difference(project(actual), expected)
        raise WriterMismatch('observed scene differs from independent structural intents',
                             edit=scene_edit(actual, expected, path, lineage or {}), evidence=dict(path=path))


def scene_edit(actual, expected, path, lineage):
    """The edit a structural scene difference at `path` names: among the committed edits of `lineage` that changed
    the node at `path` in either scene, as one of their identities or as the parent whose children they created,
    moved or erased, the latest in plan order (round_plans.latest_edit); None if no native edit changed it."""
    ids = {node['id'] for node in list(actual) + list(expected) if node['path'] == path}
    return round_plans.latest_edit({name: entry for name, entry in lineage.items()
                                    if ids & (set(entry['ids']) | set(entry['parents']))})


def account(plan, admissions, peers, commits, rounds, bindings=None):
    """Exact-once/FIFO ledger proof plus independent semantic scene and lineage.

Opaque payload equality correlates commits with accepted calls; wire decoding is
never used to compute expected contents. All later-deleted/overwritten edits
remain mandatory in the ledger. Each peer record contains separately authored
high-level effects and the original independently encoded payload.

Namespace conflicts follow the Live Edit service: a Create at an occupied
(parent, name) slot deletes the occupant's subtree, and a Move deletes either
the occupant or, for a missing parent, cycle or ancestor occupant, the mover.
A writer's rename moves its branch to the end of its parent's children, as
the host's rename and the Reorder it compiles do, and a reorder reverses its
parent's prim children and sets the parent's reorder field; generated plans
give either its parent alone in its round (_draw). A field edit's value
is the plan's typed value, whatever its family.
Every lineage entry records an outcome: reflected, or superseded when a later
edit overwrote its field or deleted its subtree, including an edit committed
after its writer's branch was already deleted (the service skips it or keeps it
as orphan data).

FIFO is by position in each writer's planned edits. `bindings`, the native
writers' wire identities by position, attributes an unexplained commit to its
writer's next planned edit; a mismatch names the edit it concerns when known.
"""
    planned = {op['edit']: op for writer in plan['writers'] for op in writer[:rounds]}
    if len(admissions) != len(planned) or {a['edit'] for a in admissions} != set(planned):
        first, evidence = round_plans.admission_discrepancy(plan, planned, admissions)
        raise WriterMismatch('missing or duplicate structural admission', edit=first, evidence=evidence)
    position = {op['edit']: index for writer in plan['writers'] for index, op in enumerate(writer)}
    column = {op['edit']: index for index, writer in enumerate(plan['writers']) for op in writer}
    records, accepted = {}, set()
    for admission in admissions:
        op = planned[admission['edit']]
        key = op['writer'], admission['accepted_id']
        if type(key[1]) is not int or key[1] <= 0 or key in accepted:
            raise WriterMismatch('invalid/repeated accepted ID', edit=op['edit'])
        accepted.add(key)
        payload = admission['bytes']
        if not payload or payload in records:
            raise WriterMismatch('ambiguous admitted payload', edit=op['edit'])
        records[payload] = (op, admission)
    for peer in peers:
        if not peer['bytes'] or peer['bytes'] in records:
            raise WriterMismatch('ambiguous peer payload')
        records[peer['bytes']] = (None, peer)

    def next_edit(wire_writer):
        """The next planned edit of the native writer whose wire identity is `wire_writer`, if bound."""
        if bindings is None or wire_writer not in bindings:
            return None
        ops = plan['writers'][bindings.index(wire_writer)][:rounds]
        index = last.get(ops[0]['writer'], -1) + 1 if ops else 0
        return ops[index]['edit'] if index < len(ops) else None
    nodes = {1: dict(id=1, parent=0, type=7, path='/', fields={'primChildren': tokens('P', 'Q')}),
             2: dict(id=2, parent=1, type=6, path='/P', fields={'primChildren': tokens('Child')}),
             3: dict(id=3, parent=2, type=6, path='/P/Child', fields={'properties': tokens('value')}),
             4: dict(id=4, parent=3, type=1, path='/P/Child.value', fields={'value': value('int32', 10)}),
             5: dict(id=5, parent=1, type=6, path='/Q', fields={'properties': tokens('other')}),
             6: dict(id=6, parent=5, type=1, path='/Q.other', fields={'value': value('int32', 20)})}
    used, branches, seen, writers, last, lineage, field_owner = set(nodes), {}, set(), {}, {}, {}, {}

    def own(identity, key, edit):
        """`edit` sets field `key` of `identity`, superseding the field's earlier writer."""
        field = identity, key
        if field in field_owner:
            lineage[field_owner[field]]['superseded_by'] = edit
        field_owner[field] = edit

    def fresh(identity, edit=None):
        if type(identity) is not int or not 1 < identity < 2**64 or identity in used:
            raise WriterMismatch('reused or invalid graph identity', edit=edit)
        used.add(identity)

    def child_names(parent):
        return nodes[parent]['fields'].setdefault('primChildren', tokens())['value']

    def subtree(identity):
        ids = {identity}
        while True:
            added = {n['id'] for n in nodes.values() if n['parent'] in ids} - ids
            if not added:
                return ids
            ids |= added

    def occupied(parent, name, mover=None):
        """The service resolves a (parent, name) slot to its lowest live prim identity."""
        prefix = '' if parent == 1 else nodes[parent]['path']
        return min((i for i, n in nodes.items() if i != mover and n['parent'] == parent and n['type'] == 6
                    and n['path'] == prefix + '/' + name), default=None)

    def displace(identity, edit):
        """Service cascade delete; earlier edits owning the subtree are superseded."""
        doomed = subtree(identity)
        child_names(nodes[identity]['parent']).remove(nodes[identity]['path'].rsplit('/', 1)[1])
        for target in doomed:
            del nodes[target]
        for branch in branches.values():
            if branch['ids'][''] in doomed:
                branch.setdefault('displaced_by', edit)
        for other, entry in lineage.items():
            if other != edit and set(entry['ids']) & doomed:
                entry.setdefault('superseded_by', edit)

    def relocate(identity, parent, name, edit):
        old = nodes[identity]['path']
        old_parent = nodes[identity]['parent']
        old_name = old.rsplit('/', 1)[1]
        if old_parent == parent:
            if old_name == name:
                return True
            # A rename onto an occupied name deletes the occupant, a sibling, with its subtree first.
            occupant = occupied(parent, name, identity)
            if occupant is not None:
                displace(occupant, edit)
            names = child_names(parent)
            names[names.index(old_name)] = name
        else:
            occupant = occupied(parent, name, identity) if parent in nodes else None
            if parent not in nodes or parent in subtree(identity) or (
                    occupant is not None and identity in subtree(occupant)):
                displace(identity, edit)
                return False
            if occupant is not None:
                displace(occupant, edit)
            child_names(old_parent).remove(old_name)
            child_names(parent).append(name)
        new = ('' if parent == 1 else nodes[parent]['path']) + '/' + name
        for target in subtree(identity):
            nodes[target]['path'] = new + nodes[target]['path'][len(old):]
        nodes[identity]['parent'] = parent
        return True

    sequence = 1
    for commit in commits:
        if commit['sequence'] != sequence + 1:
            # The gap names the first admitted edit, in plan order, that the ledger lacks; which commit follows the
            # gap depends on the OS-scheduled commit order within a round.
            committed = {c['bytes'] for c in commits}
            raise WriterMismatch('missing structural sequence',
                                 edit=round_plans.first_edit(plan, {op['edit'] for payload, (op, _) in records.items()
                                                                    if op and payload not in committed}),
                                 evidence=dict(gap=commit['sequence'] - sequence))
        sequence = commit['sequence']
        payload = commit['bytes']
        if payload not in records or payload in seen:
            raise WriterMismatch('unexplained or repeated structural commit', edit=next_edit(commit['writer']))
        seen.add(payload)
        op, admission = records[payload]
        if op is None:
            if commit['writer'] != 'independent-peer':
                raise WriterMismatch('peer writer mismatch')
            for effect in admission['effects']:
                kind = effect['kind']
                if kind not in ('move', 'create', 'field'):
                    raise WriterMismatch('unknown peer effect')
                if kind == 'create':
                    fresh(effect['id'])
                target = effect['parent'] if kind == 'create' else effect['id']
                if target not in nodes:
                    continue  # The service skips a missing target and keeps field data as orphans.
                if kind == 'move':
                    relocate(effect['id'], effect['parent'], effect['name'], payload)
                elif kind == 'create':
                    identity, parent, name = effect['id'], effect['parent'], effect['name']
                    occupant = occupied(parent, name)
                    if occupant is not None:
                        displace(occupant, payload)
                    nodes[identity] = dict(id=identity, parent=parent, type=6,
                        path=('' if parent == 1 else nodes[parent]['path']) + '/' + name,
                        fields={'marker': value('int32', effect['value'])})
                    child_names(parent).append(name)
                else:
                    nodes[effect['id']]['fields'][effect['key']] = value('int32', effect['value'])
            continue
        writer, edit, kind = op['writer'], op['edit'], op['kind']
        if position[edit] != last.get(writer, -1) + 1:
            raise WriterMismatch('per-writer structural FIFO violated', edit=edit)
        last[writer] = position[edit]
        actual_writer = commit['writer']
        if not actual_writer or actual_writer == 'independent-peer' or writers.setdefault(writer, actual_writer) != actual_writer:
            raise WriterMismatch('unstable structural writer identity', edit=edit)
        # `parents` holds the nodes whose children the edit changed; `round` and `writer` its plan position.
        entry = lineage[edit] = dict(sequence=sequence, kind=kind, accepted_id=admission['accepted_id'], ids=[],
                                     parents=[], round=position[edit], writer=column[edit])
        if kind in NEEDS_BRANCH and writer not in branches:
            raise WriterMismatch('operation on a missing writer branch', edit=edit)
        if kind in NEEDS_BRANCH and 'displaced_by' in branches[writer]:
            # The service commits the edit as orphan data or a skipped command.
            entry.update(outcome='superseded', superseded_by=branches[writer]['displaced_by'])
            if kind == 'erase':
                branches.pop(writer)
        elif kind == 'create':
            if writer in branches:
                raise WriterMismatch('creation before previous branch erased', edit=edit)
            captured = admission['captured_prefix']
            specs = template(op['shape'], captured, op['value'])
            ids = admission['ids']
            if set(ids) != set(specs) or len(set(ids.values())) != len(ids):
                raise WriterMismatch('incomplete or aliased constructor identity binding', edit=edit)
            occupant = occupied(op['parent'], op['name'])
            if occupant is not None:
                displace(occupant, edit)
            prefix = nodes[op['parent']]['path'] + '/' + op['name']
            for suffix, (spec_type, parent_suffix, fields) in specs.items():
                identity = ids[suffix]
                fresh(identity, edit)
                nodes[identity] = dict(id=identity, parent=op['parent'] if parent_suffix is None else ids[parent_suffix],
                    type=spec_type, path=prefix + suffix, fields=copy.deepcopy(fields))
            child_names(op['parent']).append(op['name'])
            branches[writer] = dict(ids=ids, shape=op['shape'], parent=op['parent'], name=op['name'])
            entry.update(ids=list(ids.values()), parents=[op['parent']])
        elif kind in ('field', 'shared'):
            identity = 2 if kind == 'shared' else leaf(branches[writer])
            key = 'shared' if kind == 'shared' else 'default'
            nodes[identity]['fields'][key] = copy.deepcopy(op['value'])
            own(identity, key, edit)
            entry.update(ids=[identity], field=key, value=op['value'])
        elif kind == 'move':
            branch = branches[writer]
            if branch['parent'] == op['parent']:
                # A same-parent rename compiles a Reorder, outside this profile.
                raise WriterMismatch('writer move must change parent', edit=edit)
            previous = branch['parent']
            if not relocate(branch['ids'][''], op['parent'], op['name'], edit):
                entry.update(ids=sorted(set(branch['ids'].values())), parents=[previous])
                continue
            branch.update(parent=op['parent'], name=op['name'])
            entry.update(ids=sorted(subtree(branch['ids'][''])), parents=sorted({previous, op['parent']}))
        elif kind in ('rename', 'reorder'):
            branch = branches[writer]
            if branch['parent'] != op['parent']:
                raise WriterMismatch(f'writer {kind} names a parent its branch is not under', edit=edit)
            parent = op['parent']
            if kind == 'rename':
                # The service's rename keeps the branch's place, but the host's namespace rename appends the renamed
                # prim to its parent's children, and the writer's Reorder lists that order, so the branch ends last.
                # Its name is unique, so it displaces none.
                relocate(branch['ids'][''], parent, op['name'], edit)
                names = child_names(parent)
                names.remove(op['name'])
                names.append(op['name'])
                branch['name'] = op['name']
                siblings = [i for i, n in nodes.items() if n['parent'] == parent]
                entry.update(ids=sorted(set(subtree(branch['ids'][''])) | set(siblings)), parents=[parent])
            else:
                # The Reorder lists the parent's prim children reversed, and no other edit of the round changes
                # them, so it reverses the children the service holds; its field makes its bytes distinct.
                child_names(parent).reverse()
                nodes[parent]['fields'][REORDER_FIELD] = value('int32', op['value'])
                own(parent, REORDER_FIELD, edit)
                # Every child's place among its siblings changes with the parent's list.
                children = [i for i, n in nodes.items() if n['parent'] == parent]
                entry.update(ids=sorted([parent, *children]), parents=[parent], field=REORDER_FIELD,
                             value=op['value'])
        elif kind == 'erase':
            branch = branches.pop(writer)
            doomed = subtree(branch['ids'][''])
            child_names(branch['parent']).remove(branch['name'])
            entry.update(ids=sorted(doomed), parents=[branch['parent']])
            for identity in doomed:
                del nodes[identity]
        else:
            raise WriterMismatch('unknown structural intent', edit=edit)
    if seen != set(records) or len(set(writers.values())) != (plan['client_count'] if rounds else 0):
        lost = [op['edit'] if op else 'peer' for payload, (op, _) in records.items() if payload not in seen]
        raise WriterMismatch('lost admitted edit/peer effect or shared writer identity',
                             edit=round_plans.first_edit(plan, {name for name in lost if name != 'peer'}),
                             evidence=dict(missing=sorted(lost)))
    for entry in lineage.values():
        entry['live_ids'] = [i for i in entry['ids'] if i in nodes]
        entry['deleted_ids'] = [i for i in entry['ids'] if i not in nodes]
        entry.setdefault('outcome', 'superseded' if 'superseded_by' in entry else 'reflected')
    return dict(scene=project(nodes.values()), branches=branches, lineage=lineage, used_ids=sorted(used))
