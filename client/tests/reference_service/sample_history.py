# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent sample intent/accounting oracle; no wire, USD or client imports.

Repeated equal payloads are matched by prebound writer identity and FIFO
occurrence, never by value uniqueness. Seeds reproduce inputs and gate choices,
not OS scheduling. The workload admits one operation per writer per round.
"""
import copy
import hashlib
import json
import math
import random
import struct


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


def verify_scene(actual, expected):
    projected = [{k: node[k] for k in ('id', 'parent', 'type', 'path', 'fields')}
                 for node in sorted(actual, key=lambda n: n['id'])]
    if json.dumps(projected, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise AssertionError('scene differs from independent sample intents')


def sample_keys(scene, section):
    """The time bits of a typed scene node's sample keys."""
    for node in scene:
        if node['id'] == section:
            return [item['time_bits'] for item in node['fields'].get('timeSamples', {}).get('value', [])]
    return []


def verify_effect(op, observed, view=()):
    """Check an independently decoded admission against its authored intent.

    Callers decode wire bytes; this oracle never derives expected semantics from
    those bytes. Even an edit later overwritten must encode its complete intent.
    `view` holds the time bits of the keys in the writer's map for the op's
    section when it compiled: a whole-map edit is sample records that erase each
    of those keys whose bits differ from its own, then set its own.
    """
    scalar = None if op['value'] is None else dict(type='int32', value=op['value'])
    fields, samples = [], []
    if op['kind'] == 'sample-map':
        samples = [dict(section=op['section'], time_bits=key, value=None)
                   for key in sorted(view, key=time) if key != op['time_bits']]
        samples.append(dict(section=op['section'], time_bits=op['time_bits'], value=scalar))
    elif op['kind'] in ('sample', 'erase-sample'):
        samples = [dict(section=op['section'], time_bits=op['time_bits'], value=scalar)]
    else:
        raise AssertionError('unsupported authored sample operation')
    wanted = dict(is_diff=False, base_version=0, source_format='', commands=0, fields=fields, samples=samples)
    if json.dumps(observed, sort_keys=True, allow_nan=False) != json.dumps(wanted, sort_keys=True, allow_nan=False):
        raise AssertionError('admitted wire effect differs from planned sample intent')


def account(plan, bindings, admissions, peers, commits, rounds):
    if not 0 <= rounds <= plan['rounds']:
        raise AssertionError('invalid accounted prefix')
    if (len(bindings) != plan['client_count'] or len(set(bindings)) != len(bindings) or
            any(not isinstance(w, str) or not w or w == 'independent-peer' for w in bindings)):
        raise AssertionError('missing or aliased prebound writer identities')
    planned = {op['edit']: op for writer in plan['writers'] for op in writer[:rounds]}
    if len(admissions) != len(planned) or {a['edit'] for a in admissions} != set(planned):
        raise AssertionError('missing or duplicate sample admission')
    by_edit, accepted = {}, set()
    for admission in admissions:
        op = planned[admission['edit']]
        key = op['writer'], admission['accepted_id']
        if (type(key[1]) is not int or key[1] <= 0 or key in accepted or
                type(admission['local_ordinal']) is not int or admission['local_ordinal'] != op['ordinal'] + 1 or
                admission['writer'] != bindings[op['writer']] or not isinstance(admission['bytes'], str) or
                not admission['bytes']):
            raise AssertionError('invalid sample admission identity')
        accepted.add(key); by_edit[op['edit']] = admission
    peer_ids = {p['edit'] for p in peers}
    native_ids = {op['edit'] for writer in plan['writers'] for op in writer}
    if (len(peer_ids) != len(peers) or peer_ids.intersection(native_ids) or
            any(not isinstance(p['bytes'], str) or not p['bytes'] for p in peers)):
        raise AssertionError('ambiguous peer sample history')
    indices, seen_peers, lineage, maps = [0]*len(bindings), set(), {}, {4: {}, 6: {}}
    # Every writer compiles a round's edit against the state at the end of the previous round.
    views = {}
    sequence = 1
    for commit in commits:
        if type(commit['sequence']) is not int or commit['sequence'] != sequence + 1:
            raise AssertionError('missing sample sequence')
        sequence = commit['sequence']
        if commit['writer'] == 'independent-peer':
            if len(seen_peers) >= len(peers):
                raise AssertionError('extra peer sample occurrence')
            op = peers[len(seen_peers)]
            if op['bytes'] != commit['bytes']:
                raise AssertionError('missing or repeated peer sample')
            if indices != [op['after_round'] + 1]*len(bindings):
                raise AssertionError('peer sample outside recorded rotation boundary')
            seen_peers.add(op['edit'])
        else:
            if commit['writer'] not in bindings:
                raise AssertionError('unexpected wire writer')
            writer = bindings.index(commit['writer']); ordinal = indices[writer]
            if ordinal >= rounds:
                raise AssertionError('extra committed sample occurrence')
            if ordinal != min(indices):
                raise AssertionError('sample commit crossed round admission barrier')
            op = plan['writers'][writer][ordinal]
            views.setdefault(ordinal, {key: {stamp: item['time_bits'] for stamp, item in value.items()}
                                       for key, value in maps.items()})
            admission = by_edit[op['edit']]
            if commit['bytes'] != admission['bytes']:
                raise AssertionError('sample FIFO or original bytes changed')
            indices[writer] += 1
        edit, kind, section = op['edit'], op['kind'], op['section']
        stamp = time(op['time_bits'])
        if section not in maps or kind not in ('sample', 'sample-map', 'erase-sample'):
            raise AssertionError('unsupported sample intent')
        if kind != 'erase-sample' and (type(op['value']) is not int or not -(2**31) <= op['value'] < 2**31):
            raise AssertionError('invalid sample value')
        current = maps[section]
        entry = lineage[edit] = dict(sequence=sequence, kind=kind, section=section, time_bits=op['time_bits'],
                                     value=op['value'], displaced=[], outcome='no_effect')
        # A whole-map edit erases the keys its writer saw, except one of identical bits; a key a peer added since
        # survives.
        seen = views[indices[writer] - 1][section] if kind == 'sample-map' and commit['writer'] != 'independent-peer' else {}
        doomed = [key for key in current if key == stamp or key in seen]
        existing = current.get(stamp)
        if stamp in seen and seen[stamp] != op['time_bits']:
            existing = None  # Erased before the set, so the set authors the key's bits.
        for key in doomed:
            previous = current.pop(key)
            lineage[previous['edit']].update(outcome='superseded' if kind != 'erase-sample' else 'erased', superseded_by=edit)
            entry['displaced'].append(previous['edit'])
        if kind != 'erase-sample':
            # A sample record at an equivalent existing key keeps that key's bits.
            key_bits = existing['time_bits'] if existing else op['time_bits']
            current[stamp] = dict(time_bits=key_bits, value=op['value'], edit=edit)
            entry['outcome'] = 'reflected'
        elif doomed:
            entry['outcome'] = 'removed'
    if indices != [rounds]*len(bindings) or seen_peers != {p['edit'] for p in peers}:
        raise AssertionError('admitted sample occurrence missing from ledger')
    scene = initial_scene()
    for node in scene:
        current = maps.get(node['id'])
        if current:
            node['fields']['timeSamples'] = dict(type='time_samples', value=[
                dict(time_bits=item['time_bits'], value=dict(type='int32', value=item['value']))
                for _, item in sorted(current.items())])
    return dict(scene=copy.deepcopy(scene), lineage=lineage, native_commits=sum(indices), peer_commits=len(seen_peers))
