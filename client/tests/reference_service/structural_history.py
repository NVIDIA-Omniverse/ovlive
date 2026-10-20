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


def generate(seed, clients=4):
    if not 2 <= clients <= 8:
        raise ValueError('bounded structural profile needs 2..8 writers')
    seeds = [int.from_bytes(hashlib.sha256(f'structural-writer-v1:{seed}:{i}'.encode()).digest()[:8], 'big')
             for i in range(clients)]
    shapes = ('usd-create', 'usd-variants', 'usd-targets')
    writers = []
    kinds = ('create', 'field', 'move', 'shared', 'erase', 'create', 'field', 'move', 'shared', 'erase', 'create', 'field')
    for writer, writer_seed in enumerate(seeds):
        rng, ops, cycle, parent = random.Random(writer_seed), [], 0, 2
        name = f'Writer{writer}_{rng.randrange(10000)}'
        for ordinal, kind in enumerate(kinds):
            if kind == 'move':
                # Every move changes parent. A cross-parent move appends and
                # compiles no Reorder, so the workload stays within the
                # service's verbatim Reorder rule; unique names keep it free of
                # random collisions, and permanent sentinel children remain.
                parent = 5 if parent == 2 else 2
                name = f'Moved{writer}_{ordinal}_{rng.randrange(10000)}'
            op = dict(writer=writer, ordinal=ordinal, edit=f'w{writer}:{ordinal}', kind=kind,
                      parent=parent, name=name, value=(writer + 1) * 100000 + ordinal * 100 + rng.randrange(100))
            if kind == 'create':
                op['shape'] = shapes[(writer + cycle) % len(shapes)]
                cycle += 1
            ops.append(op)
        writers.append(ops)
    rng = random.Random(seed)
    releases = [rng.sample(range(clients), clients) for _ in kinds]
    return dict(schema_version=1, scenario='private-attachment.structural-v1', master_seed=seed,
                writer_seeds=seeds, client_count=clients, rounds=len(kinds), writers=writers,
                release_orders=releases, rotation_after=[5, 9],
                replay_scope='exact generated operations and named gates; observed OS/gRPC order may differ')


def leaf(branch):
    return branch['ids'][{'usd-create': '/Child.amount', 'usd-variants': '{look=high-low}.amount',
                          'usd-targets': '.amount'}[branch['shape']]]


def project(nodes):
    keys = ('id', 'parent', 'type', 'path', 'fields')
    return [{key: copy.deepcopy(node[key]) for key in keys} for node in sorted(nodes, key=lambda n: n['id'])]


def verify_scene(actual, expected):
    # JSON's typed scalar spelling avoids Python's True == 1 equivalence.
    if json.dumps(project(actual), sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise AssertionError('observed scene differs from independent structural intents')


def account(plan, admissions, peers, commits, rounds):
    """Exact-once/FIFO ledger proof plus independent semantic scene and lineage.

Opaque payload equality correlates commits with accepted calls; wire decoding is
never used to compute expected contents. All later-deleted/overwritten edits
remain mandatory in the ledger. Each peer record contains separately authored
high-level effects and the original independently encoded payload.

Namespace conflicts follow the Live Edit service: a Create at an occupied
(parent, name) slot deletes the occupant's subtree, and a Move deletes either
the occupant or, for a missing parent, cycle or ancestor occupant, the mover.
Every lineage entry records an outcome: reflected, or superseded when a later
edit overwrote its field or deleted its subtree, including an edit committed
after its writer's branch was already deleted (the service skips it or keeps it
as orphan data).
"""
    planned = {op['edit']: op for writer in plan['writers'] for op in writer[:rounds]}
    if len(admissions) != len(planned) or {a['edit'] for a in admissions} != set(planned):
        raise AssertionError('missing or duplicate structural admission')
    records, accepted = {}, set()
    for admission in admissions:
        op = planned[admission['edit']]
        key = op['writer'], admission['accepted_id']
        if type(key[1]) is not int or key[1] <= 0 or key in accepted:
            raise AssertionError('invalid/repeated accepted ID')
        accepted.add(key)
        payload = admission['bytes']
        if not payload or payload in records:
            raise AssertionError('ambiguous admitted payload')
        records[payload] = (op, admission)
    for peer in peers:
        if not peer['bytes'] or peer['bytes'] in records:
            raise AssertionError('ambiguous peer payload')
        records[peer['bytes']] = (None, peer)
    nodes = {1: dict(id=1, parent=0, type=7, path='/', fields={'primChildren': tokens('P', 'Q')}),
             2: dict(id=2, parent=1, type=6, path='/P', fields={'primChildren': tokens('Child')}),
             3: dict(id=3, parent=2, type=6, path='/P/Child', fields={'properties': tokens('value')}),
             4: dict(id=4, parent=3, type=1, path='/P/Child.value', fields={'value': value('int32', 10)}),
             5: dict(id=5, parent=1, type=6, path='/Q', fields={'properties': tokens('other')}),
             6: dict(id=6, parent=5, type=1, path='/Q.other', fields={'value': value('int32', 20)})}
    used, branches, seen, writers, last, lineage, field_owner = set(nodes), {}, set(), {}, {}, {}, {}

    def fresh(identity):
        if type(identity) is not int or not 1 < identity < 2**64 or identity in used:
            raise AssertionError('reused or invalid graph identity')
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
            if occupied(parent, name, identity) is not None:
                raise AssertionError('service keeps duplicate sibling names after a rename; not modeled')
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
            raise AssertionError('missing structural sequence')
        sequence = commit['sequence']
        payload = commit['bytes']
        if payload not in records or payload in seen:
            raise AssertionError('unexplained or repeated structural commit')
        seen.add(payload)
        op, admission = records[payload]
        if op is None:
            if commit['writer'] != 'independent-peer':
                raise AssertionError('peer writer mismatch')
            for effect in admission['effects']:
                kind = effect['kind']
                if kind not in ('move', 'create', 'field'):
                    raise AssertionError('unknown peer effect')
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
        if op['ordinal'] != last.get(writer, -1) + 1:
            raise AssertionError('per-writer structural FIFO violated')
        last[writer] = op['ordinal']
        actual_writer = commit['writer']
        if not actual_writer or actual_writer == 'independent-peer' or writers.setdefault(writer, actual_writer) != actual_writer:
            raise AssertionError('unstable structural writer identity')
        entry = lineage[edit] = dict(sequence=sequence, kind=kind, accepted_id=admission['accepted_id'], ids=[])
        if kind != 'create' and writer not in branches:
            raise AssertionError('operation on a missing writer branch')
        if kind in ('field', 'move', 'erase') and 'displaced_by' in branches[writer]:
            # The service commits the edit as orphan data or a skipped command.
            entry.update(outcome='superseded', superseded_by=branches[writer]['displaced_by'])
            if kind == 'erase':
                branches.pop(writer)
        elif kind == 'create':
            if writer in branches:
                raise AssertionError('creation before previous branch erased')
            captured = admission['captured_prefix']
            specs = template(op['shape'], captured, op['value'])
            ids = admission['ids']
            if set(ids) != set(specs) or len(set(ids.values())) != len(ids):
                raise AssertionError('incomplete or aliased constructor identity binding')
            occupant = occupied(op['parent'], op['name'])
            if occupant is not None:
                displace(occupant, edit)
            prefix = nodes[op['parent']]['path'] + '/' + op['name']
            for suffix, (spec_type, parent_suffix, fields) in specs.items():
                identity = ids[suffix]
                fresh(identity)
                nodes[identity] = dict(id=identity, parent=op['parent'] if parent_suffix is None else ids[parent_suffix],
                    type=spec_type, path=prefix + suffix, fields=copy.deepcopy(fields))
            child_names(op['parent']).append(op['name'])
            branches[writer] = dict(ids=ids, shape=op['shape'], parent=op['parent'], name=op['name'])
            entry['ids'] = list(ids.values())
        elif kind in ('field', 'shared'):
            identity = 2 if kind == 'shared' else leaf(branches[writer])
            key = 'shared' if kind == 'shared' else 'default'
            nodes[identity]['fields'][key] = value('int32', op['value'])
            field = identity, key
            if field in field_owner:
                lineage[field_owner[field]]['superseded_by'] = edit
            field_owner[field] = edit
            entry.update(ids=[identity], field=key, value=op['value'])
        elif kind == 'move':
            branch = branches[writer]
            if branch['parent'] == op['parent']:
                # A same-parent rename compiles a Reorder, outside this profile.
                raise AssertionError('writer move must change parent')
            if not relocate(branch['ids'][''], op['parent'], op['name'], edit):
                entry['ids'] = sorted(set(branch['ids'].values()))
                continue
            branch.update(parent=op['parent'], name=op['name'])
            entry['ids'] = sorted(subtree(branch['ids']['']))
        elif kind == 'erase':
            branch = branches.pop(writer)
            doomed = subtree(branch['ids'][''])
            child_names(branch['parent']).remove(branch['name'])
            entry['ids'] = sorted(doomed)
            for identity in doomed:
                del nodes[identity]
        else:
            raise AssertionError('unknown structural intent')
    if seen != set(records) or len(set(writers.values())) != (plan['client_count'] if rounds else 0):
        raise AssertionError('lost admitted edit/peer effect or shared writer identity')
    for entry in lineage.values():
        entry['live_ids'] = [i for i in entry['ids'] if i in nodes]
        entry['deleted_ids'] = [i for i in entry['ids'] if i not in nodes]
        entry.setdefault('outcome', 'superseded' if 'superseded_by' in entry else 'reflected')
    return dict(scene=project(nodes.values()), branches=branches, lineage=lineage, used_ids=sorted(used))
