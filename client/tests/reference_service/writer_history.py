# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seeded root-field workload and admission/commit accounting, without client code.

Replaying this plan reproduces inputs, not an OS/gRPC schedule. Conflicts follow
the independently observed service commit order; unique values identify edits
only in synthetic fixtures, never in production protocol metadata.
"""
import hashlib
import json
import random
from collections import Counter


class WriterMismatch(AssertionError):
    """Stable semantic identity, independent of trial paths and wire writer UUIDs."""
    def __init__(self, reason, **details):
        self.signature = dict(reason=reason, **details)
        super().__init__(json.dumps(self.signature, sort_keys=True))


def generate(master_seed, clients=8, steps=16):
    if not 2 <= clients <= 32 or not 2 <= steps <= 16:
        raise ValueError('bounded writer profile requires 2..32 clients and 2..16 steps')
    seeds = [int.from_bytes(hashlib.sha256(f'public-writer-v1:{master_seed}:{i}'.encode()).digest()[:8], 'big')
             for i in range(clients)]
    writers = []
    for writer, seed in enumerate(seeds):
        rng = random.Random(seed)
        writers.append([{'writer': writer, 'ordinal': step, 'edit': f'w{writer}:{step}',
                         'field': f'owned_{writer}_{step}' if step % 2 == 0 else f'shared_{rng.randrange(3)}',
                         'value': (writer + 1) * 1000000 + step * 1024 + rng.randrange(1024)}
                        for step in range(steps)])
    release = list(range(clients))
    random.Random(master_seed).shuffle(release)
    return {'schema_version': 1, 'scenario': 'public-plugin.root-fields.v1', 'master_seed': master_seed,
            'writer_seeds': seeds, 'client_count': clients, 'writers': writers, 'release_order': release,
            'replay_scope': 'exact generated inputs; observed OS/gRPC commit order may differ'}


def validate(plan):
    """Bounded root-field replay: only delete operations, preserving their origins.

    Keep every original writer nonempty. Neither this profile nor its reducer
    changes values, populations, or the simultaneous-first-RPC hold recipe.
    """
    allowed = {'schema_version', 'scenario', 'master_seed', 'writer_seeds', 'client_count',
               'writers', 'release_order', 'replay_scope', 'initial_fixture_sha256', 'candidate',
               'source_fingerprint', 'reference_identity', 'sanitizer_runtime', 'test_service_fault'}
    if (not isinstance(plan, dict) or set(plan) - allowed or type(plan.get('schema_version')) is not int or
            plan['schema_version'] != 1 or plan.get('scenario') != 'public-plugin.root-fields.v1' or
            type(plan.get('master_seed')) is not int or not 0 <= plan['master_seed'] < 2**64 or
            type(plan.get('client_count')) is not int or not 2 <= plan['client_count'] <= 32):
        raise ValueError('invalid root-field writer profile')
    if 'test_service_fault' in plan and plan['test_service_fault'] != 'corrupt-owned-0-2':
        raise ValueError('unknown test-only service defect')
    original = generate(plan['master_seed'], plan['client_count'], 16)
    if (plan.get('writer_seeds') != original['writer_seeds'] or
            plan.get('replay_scope') != original['replay_scope'] or
            not isinstance(plan.get('release_order'), list) or
            any(type(i) is not int for i in plan['release_order']) or
            plan['release_order'] != original['release_order'] or
            not isinstance(plan.get('writers'), list) or len(plan['writers']) != plan['client_count']):
        raise ValueError('invalid writer seeds, release order or population')
    for index, writer in enumerate(plan['writers']):
        if not isinstance(writer, list) or not 1 <= len(writer) <= 16:
            raise ValueError('each retained writer requires 1..16 edits')
        previous = -1
        for op in writer:
            if (not isinstance(op, dict) or type(op.get('ordinal')) is not int or
                    not previous < op['ordinal'] < 16 or type(op.get('writer')) is not int or
                    type(op.get('value')) is not int or op != original['writers'][index][op['ordinal']]):
                raise ValueError('writer edit differs from its seeded origin or FIFO order')
            previous = op['ordinal']
    return plan


def read_json(path, limit=2 * 1024 * 1024):
    """Reject ambiguous or unbounded replay JSON before any worker starts."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate replay key')
            result[key] = value
        return result
    with path.open('rb') as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError('writer replay exceeds byte budget')
    def invalid(value):
        raise ValueError('nonfinite replay value')
    try:
        return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)
    except (RecursionError, UnicodeError) as error:
        raise ValueError('invalid writer replay encoding or depth') from error


def load(path):
    return validate(read_json(path))


def verify_provenance(plan, *, source, candidate, reference, sanitizer, fixture):
    validate(plan)
    actual = [plan.get(key) for key in ('source_fingerprint', 'candidate', 'reference_identity',
                                      'sanitizer_runtime', 'initial_fixture_sha256')]
    expected = [source, candidate, reference, sanitizer, fixture]
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('writer replay source, candidate, dependencies, sanitizer or fixture identity differs')
    return plan


def account(plan, admissions, commits, *, base_sequence=1):
    """Reject lost, duplicate, invented or FIFO-reordered edits, including losers.

commits: decoded single-field updates with sequence and actual wire writer ID.
base_sequence: the last ledger sequence before the first accounted commit; the fixture's initial CreateLayer commit by default.
admissions: one entry per successful public call, keyed by its planned edit; plain USD authoring returns no admission ID, so identity is the planned edit.
Returns independent final values and explicit supersession lineage for every edit.
"""
    planned = {op['edit']: op for writer in plan['writers'] for op in writer}
    if len(admissions) != len(planned) or {a['edit'] for a in admissions} != set(planned):
        raise WriterMismatch('missing or duplicate successful admission',
                             missing=sorted(set(planned) - {a['edit'] for a in admissions}),
                             observed=sorted(a['edit'] for a in admissions if a['edit'] not in planned),
                             repeated=sorted(edit for edit, count in Counter(a['edit'] for a in admissions).items()
                                             if count > 1))
    fingerprints = {}
    for admission in admissions:
        op = planned[admission['edit']]
        fingerprint = (op['field'], op['value'])
        if fingerprint in fingerprints:
            raise WriterMismatch('ambiguous synthetic edit identity', edit=op['edit'])
        fingerprints[fingerprint] = op
    final, lineage, last, wire_writers, seen = {}, {}, {}, {}, set()
    successors = {op['edit']: (writer[i - 1]['edit'] if i else None)
                  for writer in plan['writers'] for i, op in enumerate(writer)}
    sequence = base_sequence
    for commit in commits:
        if commit['sequence'] != sequence + 1 or len(commit['fields']) != 1:
            raise WriterMismatch('missing sequence or unexpected batch in writer ledger',
                                 gap=commit['sequence'] - sequence, fields=commit['fields'])
        sequence = commit['sequence']
        field, value = next(iter(commit['fields'].items()))
        op = fingerprints.get((field, value))
        if op is None or op['edit'] in seen:
            raise WriterMismatch('unexplained or duplicate committed edit', field=field, value=value,
                                 edit=op['edit'] if op else None)
        seen.add(op['edit'])
        writer = op['writer']
        if successors[op['edit']] != last.get(writer):
            raise WriterMismatch('per-writer commit FIFO violated', edit=op['edit'],
                                 expected_previous=successors[op['edit']], actual_previous=last.get(writer))
        last[writer] = op['edit']
        actual = commit['writer']
        if not actual or wire_writers.setdefault(writer, actual) != actual:
            raise WriterMismatch('writer identity changed within workload', edit=op['edit'])
        if field in final:
            lineage[final[field]['edit']]['outcome'] = 'superseded'
            lineage[final[field]['edit']]['superseded_by'] = op['edit']
        final[field] = op
        lineage[op['edit']] = {'sequence': sequence, 'field': field, 'value': value, 'outcome': 'reflected'}
    if seen != set(planned):
        raise WriterMismatch('admitted edit missing from service ledger', missing=sorted(set(planned) - seen))
    if len(set(wire_writers.values())) != plan['client_count']:
        raise WriterMismatch('independent clients share a wire writer identity',
                             pairs=[[a, b] for a in sorted(wire_writers) for b in sorted(wire_writers)
                                    if a < b and wire_writers[a] == wire_writers[b]])
    return {field: op['value'] for field, op in final.items()}, lineage
