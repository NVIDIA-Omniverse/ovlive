# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent sample intent/accounting oracle; no wire, USD or client imports.

Repeated equal payloads are matched by prebound writer identity and FIFO
occurrence, never by value uniqueness. Seeds reproduce inputs and gate choices,
not OS scheduling. The workload authors one operation per writer per round, which
admits one intent unless it changes nothing in its writer's map.
"""
import copy
import hashlib
import json
import math
import random
import struct

import round_plans
from writer_history import WriterMismatch

# Named test-only defects a sample plan may carry. test_graph_transport implements the service defects, which
# corrupt or acknowledge without committing writer 0's round-6 edit, with CorruptGraphWriterService and
# corrupt_sample, the relay defect, which relays every record at time 3.0 of section 6 to its readers with the
# value -1 while the ledger keeps the uploaded bytes, with relay_sample, and the fault-schedule defect by
# forwarding the rotation after round 5 to the backend it leaves.
FAULTS = ('corrupt-sample-w0:6', 'swallow-sample-w0:6', 'misroute-rotation-5', 'relay-sample-6:3')


def bits(time):
    return struct.unpack('<Q', struct.pack('<d', time))[0]


def time(bits_value):
    if type(bits_value) is not int or not 0 <= bits_value < 2**64:
        raise AssertionError('invalid time bits')
    value = struct.unpack('<d', struct.pack('<Q', bits_value))[0]
    if math.isnan(value):
        raise AssertionError('NaN sample time')
    return value


def generate(seed, clients=4, rounds=16):
    if not 2 <= clients <= 8 or not 12 <= rounds <= 64:
        raise ValueError('sample profile requires 2..8 writers and 12..64 rounds')
    seeds = [int.from_bytes(hashlib.sha256(f'sample-writer-v1:{seed}:{i}'.encode()).digest()[:8], 'big')
             for i in range(clients)]
    writers = []
    for writer, writer_seed in enumerate(seeds):
        rng, ops = random.Random(writer_seed), []
        mandatory = [('sample', 4, -0.0, 7), ('sample', 4, 0.0, 7),
                     ('erase-sample', 4, 0.0, None), ('erase-sample', 4, 0.0, None),
                     ('sample', 4, 1.0, 3), ('sample-map', 4, 2.0, 3),
                     ('sample', 6, 3.0, 7), ('sample-map', 4, -0.0, 7)]
        for ordinal in range(rounds):
            if ordinal < len(mandatory):
                kind, section, stamp, value = mandatory[ordinal]
            else:
                kind = rng.choice(('sample', 'sample', 'sample-map', 'erase-sample'))
                section, stamp = rng.choice((4, 6)), rng.choice((-2.5, -0.0, 0.0, 1.0, 2.0, 3.0))
                value = None if kind == 'erase-sample' else rng.choice((-7, 0, 3, 7))
            ops.append(dict(writer=writer, ordinal=ordinal, edit=f'w{writer}:{ordinal}', kind=kind,
                            section=section, time_bits=bits(stamp), value=value))
        writers.append(ops)
    rng = random.Random(seed)
    return dict(schema_version=1, scenario='private-attachment.samples-v1', master_seed=seed,
                writer_seeds=seeds, client_count=clients, rounds=rounds, writers=writers,
                release_orders=[rng.sample(range(clients), clients) for _ in range(rounds)],
                rotation_after=[5, 11], replay_scope='generated inputs and named gates; observed commit order may differ')


def origin(plan):
    """The generated plan `plan` is, or was selected from (round_plans.select)."""
    return round_plans.validate_selection(plan, generate, FAULTS)


def validate(plan):
    """A generated plan or a selection of one; any selection executes, since each edit compiles against the map
    its writer holds when it is authored."""
    origin(plan)
    return plan


def initial_scene():
    def tokens(*items):
        return dict(type='tokens', value=list(items))
    def node(identity, parent, kind, path, fields):
        return dict(id=identity, parent=parent, type=kind, path=path, fields=fields)
    return [node(1, 0, 7, '/', {'primChildren': tokens('P', 'Q')}),
            node(2, 1, 6, '/P', {'primChildren': tokens('Child')}),
            node(3, 2, 6, '/P/Child', {'properties': tokens('value')}),
            node(4, 3, 1, '/P/Child.value', {'value': dict(type='int32', value=10)}),
            node(5, 1, 6, '/Q', {'properties': tokens('other')}),
            node(6, 5, 1, '/Q.other', {'value': dict(type='int32', value=20)})]


def verify_scene(actual, expected, lineage=None):
    """Raise a mismatch naming the edit that last affected the first differing sample key.

    `lineage` is account()'s lineage of `expected`. The first differing node's first differing sample time
    (or, for a node whose other fields differ, none) is the key. Its edit is, among the committed edits that
    emitted a record at that key's section and time, the one latest in plan order (round_plans.latest_edit); the
    service commit order within a round is OS-scheduled, so it never ranks them. Without a lineage, or for a key no
    committed edit emitted, the signature names none.
    """
    projected = [{k: node[k] for k in ('id', 'parent', 'type', 'path', 'fields')}
                 for node in sorted(actual, key=lambda n: n['id'])]
    if json.dumps(projected, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        path = round_plans.first_difference(projected, expected)
        raise WriterMismatch('scene differs from independent sample intents',
                             edit=scene_edit(projected, expected, path, lineage or {}), evidence=dict(path=path))


def scene_edit(actual, expected, path, lineage):
    """The edit a sample scene difference at `path` names: the latest in plan order (round_plans.latest_edit) of the
    committed edits of `lineage` that emitted a record at the node's first differing sample key, read from both
    scenes; None when the node differs elsewhere, in identity or in another field, or no committed edit emitted
    at that key."""
    def node(scene):
        return next((n for n in scene if n['path'] == path), None)
    def keys(item):
        return {time(sample['time_bits']): json.dumps(sample, sort_keys=True)
                for sample in (item or {}).get('fields', {}).get('timeSamples', {}).get('value', [])}
    left, right = node(actual), node(expected)
    others = [item and {k: v for k, v in item['fields'].items() if k != 'timeSamples'} for item in (left, right)]
    if others[0] != others[1] or (left and right and left['id'] != right['id']):
        return None
    before, after = keys(left), keys(right)
    stamp = min((t for t in set(before) | set(after) if before.get(t) != after.get(t)), default=None)
    section = (right or left)['id']

    return round_plans.latest_edit({name: entry for name, entry in lineage.items()
                                    if entry['sequence'] is not None and entry['section'] == section and
                                    any(time(key) == stamp for key in entry['emitted'])})


def sample_view(scene, section):
    """The time bits and int32 value of each of a typed scene node's sample keys."""
    for node in scene:
        if node['id'] == section:
            return {item['time_bits']: item['value']['value']
                    for item in node['fields'].get('timeSamples', {}).get('value', [])}
    return {}


def compiled_samples(op, view):
    """The (time bits, value) sample records a writer compiles for `op` against `view`, its map for the section.

    A whole-map edit erases each key whose bits differ from its own, then sets its own. Records at the op's
    time are one key of the map: a set keeps an equivalent existing key's bits unless an erase precedes it.
    The key's records are not emitted when they leave the map as it was, so an edit that changes nothing in its
    writer's map compiles no records.
    """
    if op['kind'] == 'sample-map':
        records = [(key, None) for key in sorted(view, key=time) if key != op['time_bits']]
        records.append((op['time_bits'], op['value']))
    elif op['kind'] in ('sample', 'erase-sample'):
        records = [(op['time_bits'], op['value'])]
    else:
        raise AssertionError('unsupported authored sample operation')
    stamp = time(op['time_bits'])
    group = [record for record in records if time(record[0]) == stamp]
    present = next((key for key in view if time(key) == stamp), None)
    key_bits, value = group[-1]
    if value is None:
        unchanged = present is None
    else:
        unchanged = present is not None and view[present] == value and (len(group) == 1 or present == key_bits)
    return [record for record in records if not (unchanged and time(record[0]) == stamp)]


def verify_effect(op, observed, view=None):
    """Check an independently decoded admission against its authored intent.

    Callers decode wire bytes; this oracle never derives expected semantics from
    those bytes. Even an edit later overwritten must encode its complete intent.
    `view` maps the time bits of the keys in the writer's map for the op's
    section when it compiled to their values (compiled_samples); without a view
    the bytes are a raw delta of the op's one record.
    """
    if view is None:
        records = [(op['time_bits'], op['value'])]
    else:
        records = compiled_samples(op, view)
    samples = [dict(section=op['section'], time_bits=key,
                    value=None if value is None else dict(type='int32', value=value)) for key, value in records]
    wanted = dict(is_diff=False, base_version=0, source_format='', commands=0, fields=[], samples=samples)
    if json.dumps(observed, sort_keys=True, allow_nan=False) != json.dumps(wanted, sort_keys=True, allow_nan=False):
        raise WriterMismatch('admitted wire effect differs from planned sample intent', edit=op['edit'])


def account(plan, bindings, admissions, peers, commits, rounds):
    if not 0 <= rounds <= plan['rounds']:
        raise WriterMismatch('invalid accounted prefix')
    if (len(bindings) != plan['client_count'] or len(set(bindings)) != len(bindings) or
            any(not isinstance(w, str) or not w or w == 'independent-peer' for w in bindings)):
        raise WriterMismatch('missing or aliased prebound writer identities')
    planned = {op['edit']: op for writer in plan['writers'] for op in writer[:rounds]}
    if len(admissions) != len(planned) or {a['edit'] for a in admissions} != set(planned):
        first, evidence = round_plans.admission_discrepancy(plan, planned, admissions)
        raise WriterMismatch('missing or duplicate sample admission', edit=first, evidence=evidence)
    by_edit, accepted, noops = {a['edit']: a for a in admissions}, set(), set()
    # An edit that changes nothing in its writer's map admits nothing; the others carry their writer's admission
    # ordinal among its admitted edits.
    for writer, ops in enumerate(plan['writers']):
        admitted = 0
        for op in ops[:rounds]:
            admission = by_edit[op['edit']]
            if admission['writer'] != bindings[writer]:
                raise WriterMismatch('invalid sample admission identity', edit=op['edit'])
            if admission.get('noop'):
                if admission.get('accepted_id') is not None or admission.get('bytes') is not None:
                    raise WriterMismatch('a no-op sample edit carries an admission', edit=op['edit'])
                noops.add(op['edit'])
                continue
            admitted += 1
            key = writer, admission['accepted_id']
            if (type(key[1]) is not int or key[1] <= 0 or key in accepted or
                    type(admission['local_ordinal']) is not int or admission['local_ordinal'] != admitted or
                    not isinstance(admission['bytes'], str) or not admission['bytes']):
                raise WriterMismatch('invalid sample admission identity', edit=op['edit'])
            accepted.add(key)
    peer_ids = {p['edit'] for p in peers}
    native_ids = {op['edit'] for writer in plan['writers'] for op in writer}
    if (len(peer_ids) != len(peers) or peer_ids.intersection(native_ids) or
            any(not isinstance(p['bytes'], str) or not p['bytes'] for p in peers)):
        raise WriterMismatch('ambiguous peer sample history')
    indices, seen_peers, lineage, maps = [0]*len(bindings), set(), {}, {4: {}, 6: {}}
    # Every writer compiles a round's edit against the state at the end of the previous round.
    views = {}

    def view(ordinal, section):
        """The writers' map for `section` when they compiled round `ordinal`: time bits to value."""
        snapshot = views.setdefault(ordinal, {key: {stamp: dict(time_bits=item['time_bits'], value=item['value'])
                                                    for stamp, item in value.items()} for key, value in maps.items()})
        return {item['time_bits']: item['value'] for item in snapshot[section].values()}

    def skip():
        """Steps past no-op edits once every writer has finished the round before theirs, and its peer sample."""
        while True:
            low, moved = min(indices), False
            if len(seen_peers) < len(peers) and peers[len(seen_peers)]['after_round'] + 1 == low:
                return
            for writer in range(len(bindings)):
                if indices[writer] == low < rounds and plan['writers'][writer][low]['edit'] in noops:
                    op = plan['writers'][writer][low]
                    if compiled_samples(op, view(low, op['section'])):
                        raise WriterMismatch('a sample edit that changes its writer map admitted nothing',
                                             edit=op['edit'])
                    lineage[op['edit']] = dict(sequence=None, kind=op['kind'], section=op['section'],
                                               time_bits=op['time_bits'], value=op['value'], displaced=[],
                                               outcome='no_effect', emitted=[], **order[op['edit']])
                    indices[writer] += 1
                    moved = True
            if not moved:
                return

    def lost():
        """The first admitted edit in plan order, or peer sample, whose payload the ledger lacks: each writer's
        committed payloads, in order, are a prefix of its admitted ones."""
        missing = set()
        for position, ops in enumerate(plan['writers']):
            admitted = [op for op in ops[:rounds] if op['edit'] not in noops]
            committed = [c['bytes'] for c in commits if c['writer'] == bindings[position]]
            index = next((i for i, op in enumerate(admitted)
                          if i >= len(committed) or committed[i] != by_edit[op['edit']]['bytes']), None)
            if index is not None:
                missing.add(admitted[index]['edit'])
        committed = [c['bytes'] for c in commits if c['writer'] == 'independent-peer']
        missing |= {p['edit'] for i, p in enumerate(peers) if i >= len(committed) or committed[i] != p['bytes']}
        return round_plans.first_edit(plan, missing, peer_rounds)

    # Plan order of every edit and peer sample: its round, a peer sample after the round's writers, then writer.
    order = {op['edit']: dict(round=index, writer=position) for position, ops in enumerate(plan['writers'])
             for index, op in enumerate(ops)}
    order.update({p['edit']: dict(round=p['after_round'], writer=None) for p in peers})
    peer_rounds = {p['edit']: p['after_round'] for p in peers}
    sequence = 1
    skip()
    for commit in commits:
        if type(commit['sequence']) is not int or commit['sequence'] != sequence + 1:
            # The gap names the first admitted edit, in plan order, that the ledger lacks; which commit follows the
            # gap depends on the OS-scheduled commit order within a round.
            raise WriterMismatch('missing sample sequence', edit=lost(),
                                 evidence=dict(gap=commit['sequence'] - sequence
                                               if type(commit['sequence']) is int else None))
        sequence = commit['sequence']
        if commit['writer'] == 'independent-peer':
            if len(seen_peers) >= len(peers):
                raise WriterMismatch('extra peer sample occurrence')
            op = peers[len(seen_peers)]
            if op['bytes'] != commit['bytes']:
                raise WriterMismatch('missing or repeated peer sample', edit=op['edit'])
            if indices != [op['after_round'] + 1]*len(bindings):
                raise WriterMismatch('peer sample outside recorded rotation boundary', edit=op['edit'])
            seen_peers.add(op['edit'])
        else:
            if commit['writer'] not in bindings:
                raise WriterMismatch('unexpected wire writer')
            writer = bindings.index(commit['writer']); ordinal = indices[writer]
            if ordinal >= rounds:
                raise WriterMismatch('extra committed sample occurrence')
            if ordinal != min(indices):
                raise WriterMismatch('sample commit crossed round admission barrier',
                                     edit=plan['writers'][writer][ordinal]['edit'])
            op = plan['writers'][writer][ordinal]
            records = compiled_samples(op, view(ordinal, op['section']))
            admission = by_edit[op['edit']]
            if commit['bytes'] != admission['bytes']:
                raise WriterMismatch('sample FIFO or original bytes changed', edit=op['edit'])
            indices[writer] += 1
        edit, kind, section = op['edit'], op['kind'], op['section']
        stamp = time(op['time_bits'])
        if section not in maps or kind not in ('sample', 'sample-map', 'erase-sample'):
            raise WriterMismatch('unsupported sample intent', edit=op['edit'])
        if kind != 'erase-sample' and (type(op['value']) is not int or not -(2**31) <= op['value'] < 2**31):
            raise WriterMismatch('invalid sample value', edit=op['edit'])
        current = maps[section]
        # The keys the commit carries records for: a peer's one sample, or the writer's compiled records.
        emitted = [op['time_bits']] if commit['writer'] == 'independent-peer' else [key for key, _ in records]
        entry = lineage[edit] = dict(sequence=sequence, kind=kind, section=section, time_bits=op['time_bits'],
                                     value=op['value'], displaced=[], outcome='no_effect', emitted=emitted,
                                     **order[edit])
        # A whole-map edit erases the keys its writer saw, except one of identical bits; a key a peer added since
        # survives. The edit's own key is written only when its records changed it in the writer's map; otherwise
        # whatever another writer committed there since stands.
        peer = commit['writer'] == 'independent-peer'
        own = peer or any(time(key) == stamp for key, _ in records)
        seen = views[indices[writer] - 1][section] if kind == 'sample-map' and not peer else {}
        doomed = [key for key in current if (key == stamp and own) or (key in seen and key != stamp)]
        existing = current.get(stamp)
        if stamp in seen and seen[stamp]['time_bits'] != op['time_bits']:
            existing = None  # Erased before the set, so the set authors the key's bits.
        for key in doomed:
            previous = current.pop(key)
            lineage[previous['edit']].update(outcome='superseded' if kind != 'erase-sample' else 'erased', superseded_by=edit)
            entry['displaced'].append(previous['edit'])
        if kind != 'erase-sample' and own:
            # A sample record at an equivalent existing key keeps that key's bits.
            key_bits = existing['time_bits'] if existing else op['time_bits']
            current[stamp] = dict(time_bits=key_bits, value=op['value'], edit=edit)
            entry['outcome'] = 'reflected'
        elif doomed:
            entry['outcome'] = 'removed'
        skip()
    if indices != [rounds]*len(bindings) or seen_peers != {p['edit'] for p in peers}:
        missing = {op['edit'] for position, ops in enumerate(plan['writers']) for op in ops[indices[position]:rounds]}
        missing |= {p['edit'] for p in peers} - seen_peers
        raise WriterMismatch('admitted sample occurrence missing from ledger',
                             edit=round_plans.first_edit(plan, missing, peer_rounds),
                             evidence=dict(missing=sorted(missing)))
    scene = initial_scene()
    for node in scene:
        current = maps.get(node['id'])
        if current:
            node['fields']['timeSamples'] = dict(type='time_samples', value=[
                dict(time_bits=item['time_bits'], value=dict(type='int32', value=item['value']))
                for _, item in sorted(current.items())])
    return dict(scene=copy.deepcopy(scene), lineage=lineage, native_commits=sum(indices) - len(noops),
                peer_commits=len(seen_peers))
