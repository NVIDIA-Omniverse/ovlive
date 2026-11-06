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
    """Stable semantic identity, independent of trial paths and wire writer UUIDs.

    `signature` is what a reduction must reproduce. `evidence` holds diagnostic detail that depends on the rest of
    the plan, such as every missing edit; it is reported and persisted, never compared.
    """
    def __init__(self, reason, *, evidence=None, **details):
        self.signature = dict(reason=reason, **details)
        self.evidence = evidence or {}
        report = dict(self.signature, evidence=self.evidence) if self.evidence else self.signature
        super().__init__(json.dumps(report, sort_keys=True))


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
               'source_fingerprint', 'reference_identity', 'sanitizer_runtime', 'test_service_fault',
               'coalesced_writes'}
    if (not isinstance(plan, dict) or set(plan) - allowed or type(plan.get('schema_version')) is not int or
            plan['schema_version'] != 1 or plan.get('scenario') != 'public-plugin.root-fields.v1' or
            type(plan.get('master_seed')) is not int or not 0 <= plan['master_seed'] < 2**64 or
            type(plan.get('client_count')) is not int or not 2 <= plan['client_count'] <= 32):
        raise ValueError('invalid root-field writer profile')
    if 'test_service_fault' in plan and plan['test_service_fault'] != 'corrupt-owned-0-2':
        raise ValueError('unknown test-only service defect')
    if 'coalesced_writes' in plan and plan['coalesced_writes'] is not True:
        raise ValueError('coalesced_writes is present only as true')
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


def first_edit(plan, edits):
    """The first of `edits` in plan order: by seeded step (`ordinal`, the round of a round-based plan), then by seeded
    writer. A reduction never renumbers either, whether it deletes single edits or whole writers and rounds, so a
    signature naming this edit survives deleting a later discrepancy or an earlier edit; within a step, the writers'
    commit order is OS-scheduled and never ranks. A name the plan does not hold follows every planned one, in name
    order. None if `edits` is empty."""
    rank = {op['edit']: (op['ordinal'], op['writer']) for writer in plan['writers'] for op in writer}
    return min(edits, key=lambda name: (name not in rank, rank.get(name, ()), name), default=None)


def account(plan, admissions, commits, *, base_sequence=1, coalesced=False):
    """Reject lost, duplicate, invented or FIFO-reordered edits, including losers.

commits: decoded updates with sequence and actual wire writer ID; single-field unless `coalesced`.
base_sequence: the last ledger sequence before the first accounted commit; the fixture's initial CreateLayer commit by default.
admissions: one entry per successful public call, keyed by its planned edit; plain USD authoring returns no admission ID, so identity is the planned edit.
coalesced: a writer may coalesce consecutive edits into one update, so each commit is the fields of a contiguous run
of one writer's edits, the last write of each field winning, and the runs of each writer, in commit order, cover its
edits exactly once in FIFO order.

A mismatch raises WriterMismatch. Each class is decided over the whole ledger, in a fixed order of classes, and its
signature is the reason and at most the first discrepant edit in plan order (first_edit), the least unexplained
field name or a seeded writer index; which commit the OS scheduled first within a phase never selects it. Every
other discrepancy, the gap, sequences and commit contents are evidence. A lost edit is named itself, before the FIFO
check its absence would also fail, and a sequence gap is evidence of the lost edit or, with none lost, names nothing.
Coalesced signatures name at most a field or a writer (account_runs).
Returns independent final values and explicit supersession lineage for every edit.
"""
    if coalesced:
        return account_runs(plan, admissions, commits, base_sequence)
    planned = admitted_once(plan, admissions)
    fingerprints = identities(plan)
    explained(commits, fingerprints)
    batches = [commit for commit in commits if len(commit['fields']) != 1]
    if batches:
        # Which updates a service batches together follows their commit order, so no edit names a batch.
        raise WriterMismatch('unexpected batch or empty update in writer ledger',
                             evidence=dict(commits=entries(batches)))
    edits = [fingerprints[next(iter(commit['fields'].items()))]['edit'] for commit in commits]
    repeated = {edit for edit, count in Counter(edits).items() if count > 1}
    if repeated:
        raise WriterMismatch('duplicate committed edit', edit=first_edit(plan, repeated),
                             evidence=dict(repeated=sorted(repeated)))
    missing = set(planned) - set(edits)
    gap = sequence_gap(commits, base_sequence)
    if missing:
        # A lost edit whose sequence the service consumed leaves a gap only if a later commit follows it, which the
        # OS schedules, so the gap is evidence and a gap names no edit.
        raise WriterMismatch('admitted edit missing from service ledger', edit=first_edit(plan, missing),
                             evidence=dict(missing=sorted(missing), **({'gap': gap} if gap else {})))
    if gap:
        raise WriterMismatch('missing sequence in writer ledger', evidence=gap)
    own = {}  # Each seeded writer's committed edits with their commits, in ledger order.
    for edit, commit in zip(edits, commits):
        own.setdefault(planned[edit]['writer'], []).append((edit, commit))
    previous = {op['edit']: (writer[i - 1]['edit'] if i else None)
                for writer in plan['writers'] for i, op in enumerate(writer)}
    violations = {}
    for pairs in own.values():
        last = None
        for edit, _ in pairs:
            if previous[edit] != last:
                violations[edit] = dict(expected_previous=previous[edit], actual_previous=last)
            last = edit
    if violations:
        raise WriterMismatch('per-writer commit FIFO violated', edit=first_edit(plan, violations),
                             evidence=dict(violations=dict(sorted(violations.items()))))
    wire, changed = {}, {}
    for writer, pairs in own.items():
        wire[writer] = pairs[0][1]['writer']
        seen = [commit['writer'] for _, commit in pairs]
        if not all(seen) or len(set(seen)) > 1:
            changed[writer] = [[edit, bool(commit['writer']) and commit['writer'] == wire[writer]]
                               for edit, commit in pairs]
    if changed:
        # The ledger does not say which of a writer's commits carries the wrong identity (its first may), so the
        # signature names the seeded writer, and each of its edits with whether it matches its first is evidence.
        raise WriterMismatch('writer identity changed within workload', writer=min(changed),
                             evidence=dict(writers=sorted(changed),
                                           matches_first=[changed[writer] for writer in sorted(changed)]))
    distinct_writers(wire)
    final, lineage = {}, {}
    for edit, commit in zip(edits, commits):
        op = planned[edit]
        field = op['field']
        if field in final:
            lineage[final[field]['edit']]['outcome'] = 'superseded'
            lineage[final[field]['edit']]['superseded_by'] = edit
        final[field] = op
        lineage[edit] = {'sequence': commit['sequence'], 'field': field, 'value': op['value'], 'outcome': 'reflected'}
    return {field: op['value'] for field, op in final.items()}, lineage


def admitted_once(plan, admissions):
    """The planned edits by name, after every planned edit is admitted exactly once; a mismatch names the first
    missing, repeated or unplanned admission in plan order."""
    planned = {op['edit']: op for writer in plan['writers'] for op in writer}
    names = [admission['edit'] for admission in admissions]
    if len(names) != len(planned) or set(names) != set(planned):
        missing, unplanned = set(planned) - set(names), set(names) - set(planned)
        repeated = {name for name, count in Counter(names).items() if count > 1}
        raise WriterMismatch('missing or duplicate successful admission',
                             edit=first_edit(plan, missing | repeated | unplanned),
                             evidence=dict(missing=sorted(missing), repeated=sorted(repeated),
                                           unplanned=sorted(unplanned)))
    return planned


def identities(plan):
    """Each planned edit by its synthetic (field, value) fingerprint. Edits sharing one are ambiguous, and the
    mismatch names the first of them in plan order."""
    fingerprints, ambiguous = {}, set()
    for writer in plan['writers']:
        for op in writer:
            other = fingerprints.setdefault((op['field'], op['value']), op)
            if other is not op:
                ambiguous.update((other['edit'], op['edit']))
    if ambiguous:
        raise WriterMismatch('ambiguous synthetic edit identity', edit=first_edit(plan, ambiguous),
                             evidence=dict(edits=sorted(ambiguous)))
    return fingerprints


def entries(commits):
    """Evidence of `commits`: each one's sequence and fields, without its wire writer ID."""
    return [dict(sequence=commit['sequence'], fields=commit['fields']) for commit in commits]


def explained(commits, fingerprints):
    """Every committed (field, value) item is a planned edit's. A mismatch names the least unexplained field name
    the plan authors, never a value, which a defect may vary, or a commit, whose order and grouping the OS schedules;
    a field no planned edit authors comes from the defect, so it is evidence and such a commit names nothing."""
    authored = {field for field, _ in fingerprints}
    def odd(commit):
        return [field for field, value in commit['fields'].items() if (field, value) not in fingerprints]
    fields = sorted({field for commit in commits for field in odd(commit)})
    evidence = dict(fields=fields, commits=entries(c for c in commits if odd(c)))
    planned = [field for field in fields if field in authored]
    if planned:
        raise WriterMismatch('unexplained committed field', field=planned[0], evidence=evidence)
    if fields:
        raise WriterMismatch('committed field no planned edit authors', evidence=evidence)


def sequence_gap(commits, base_sequence):
    """The first break, in ledger order, of contiguous sequences after `base_sequence` as evidence, or None."""
    expected = base_sequence
    for commit in commits:
        if commit['sequence'] != expected + 1:
            return dict(sequence=commit['sequence'], gap=commit['sequence'] - expected)
        expected = commit['sequence']
    return None


def distinct_writers(wire):
    """Each seeded writer in `wire` (writer to wire ID) has its own wire ID; a mismatch names the least seeded writer
    that shares one, which deleting another sharing writer keeps, and every sharing pair is evidence."""
    pairs = [[a, b] for a in sorted(wire) for b in sorted(wire) if a < b and wire[a] == wire[b]]
    if pairs:
        raise WriterMismatch('independent clients share a wire writer identity', writer=pairs[0][0],
                             evidence=dict(pairs=pairs))


def account_runs(plan, admissions, commits, base_sequence):
    """account() for coalesced commits: each commit is one contiguous run of one writer's edits.

    Which of a writer's blocks coalesce into one commit depends on when its window drains, which the OS schedules,
    so the edits a commit carries, its sequence and the first edit of a failing run differ between trials of one
    plan; only a content defect's field and the writers it touches do not. A mismatch therefore names the least
    unexplained field name the plan authors or, for runs that do not cover a writer's edits, the least seeded writer
    whose runs fail; the runs, sequences, any gap and fields are evidence. An update mixing writers fails the coverage
    of every writer it touches, so a defect that puts one writer's value into another's run gives the same signature
    whether the run carries other edits or not. A gap with every edit covered and an empty update name nothing.
    """
    admitted_once(plan, admissions)
    fingerprints = identities(plan)
    explained(commits, fingerprints)
    owners = [{fingerprints[item]['writer'] for item in commit['fields'].items()} for commit in commits]
    empty = [commit for commit in commits if not commit['fields']]
    if empty:
        raise WriterMismatch('empty update in writer ledger', evidence=dict(commits=entries(empty)))
    rows = {writer[0]['writer']: writer for writer in plan['writers'] if writer}
    spans, uncovered = {}, {}
    for writer, ops in rows.items():
        start = 0
        for index, commit in enumerate(commits):
            if writer not in owners[index]:
                continue
            if len(owners[index]) > 1:
                uncovered[writer] = dict(writer=writer,
                                         first_uncovered=ops[start]['edit'] if start < len(ops) else None,
                                         sequence=commit['sequence'], fields=commit['fields'],
                                         mixed=sorted(owners[index]))
                break
            # The run is the shortest one from the writer's first uncovered edit whose folded fields equal the commit.
            folded, end = {}, start
            while end < len(ops) and folded != commit['fields']:
                folded[ops[end]['field']] = ops[end]['value']
                end += 1
            if folded != commit['fields']:
                uncovered[writer] = dict(writer=writer,
                                         first_uncovered=ops[start]['edit'] if start < len(ops) else None,
                                         sequence=commit['sequence'], fields=commit['fields'])
                break
            spans[index] = writer, start, end
            start = end
        else:
            if start < len(ops):
                uncovered[writer] = dict(writer=writer, first_uncovered=ops[start]['edit'],
                                         missing=[op['edit'] for op in ops[start:]])
    gap = sequence_gap(commits, base_sequence)
    if uncovered:
        raise WriterMismatch("writer's commits do not cover its admitted edits in contiguous FIFO runs",
                             writer=min(uncovered), evidence=dict(uncovered=[uncovered[w] for w in sorted(uncovered)],
                                                                  **({'gap': gap} if gap else {})))
    if gap:
        raise WriterMismatch('missing sequence in writer ledger', evidence=gap)
    wire, changed = {}, set()
    for writer in rows:
        actual = [commit['writer'] for commit, writers in zip(commits, owners) if writers == {writer}]
        wire[writer] = actual[0]
        if any(not identity or identity != actual[0] for identity in actual):
            changed.add(writer)
    if changed:
        raise WriterMismatch('writer identity changed within workload', writer=min(changed),
                             evidence=dict(writers=sorted(changed)))
    distinct_writers(wire)
    final, lineage = {}, {}
    for index, commit in enumerate(commits):
        writer, start, end = spans[index]
        ops = rows[writer]
        for position in range(start, end):
            op = ops[position]
            later = next((ops[j] for j in range(position + 1, end) if ops[j]['field'] == op['field']), None)
            if later:
                lineage[op['edit']] = {'sequence': commit['sequence'], 'field': op['field'], 'value': op['value'],
                                       'outcome': 'coalesced', 'superseded_by': later['edit']}
                continue
            if op['field'] in final:
                lineage[final[op['field']]['edit']]['outcome'] = 'superseded'
                lineage[final[op['field']]['edit']]['superseded_by'] = op['edit']
            final[op['field']] = op
            lineage[op['edit']] = {'sequence': commit['sequence'], 'field': op['field'], 'value': op['value'],
                                   'outcome': 'reflected'}
    return {field: op['value'] for field, op in final.items()}, lineage
