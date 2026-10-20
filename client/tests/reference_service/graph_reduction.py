# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded differential-history reduction; original payloads are never overwritten.

Only a reproduced semantic discrepancy is accepted. Crashes, malformed output,
timeouts and unrelated mismatches are not substitutes for that discrepancy.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import struct
from pathlib import Path

from scene_graph_model import Model, GraphError
from scene_graph_wire import Delta, decode, encode, typed_value, GraphWireError, REAL_FAMILIES
from tools.live.test_support import run_process, write_json


@dataclass(frozen=True)
class Case:
    origin: int
    name: str
    payload: bytes

    def reference(self):
        return dict(origin=self.origin, name=self.name, sha256=hashlib.sha256(self.payload).hexdigest())


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def expected(cases):
    model, records = Model(), []
    for case in cases:
        before = model.state, model.source_format
        result = 'ok'
        try:
            model.apply(case.payload, delivered=True)
        except GraphError as error:
            result = error.kind
            if (model.state, model.source_format) != before:
                raise AssertionError('oracle rejection changed prior scene')
        records.append(dict(result=result, scene=dict(source_format=model.source_format,
            graph=[dict(node, fields={k: typed_value(v) for k, v in node['fields'].items()}) for node in model.state])))
    return records


def difference(a, b, path=()):
    """Stable first difference, with graph rows addressed by ID rather than index."""
    if canonical(a) == canonical(b):
        return None
    if type(a) is not type(b):
        return dict(path=path, kind='type', expected_type=type(a).__name__, actual_type=type(b).__name__,
                    expected=digest(a), actual=digest(b))
    if isinstance(a, dict):
        if a.keys() != b.keys():
            missing, extra = sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())
            return dict(path=path, kind='keys', missing=missing, extra=extra,
                        expected=digest({k: a[k] for k in missing}), actual=digest({k: b[k] for k in extra}))
        for key in sorted(a):
            found = difference(a[key], b[key], path + (key,))
            if found:
                return found
    elif isinstance(a, list):
        identity = 'id' if path == ('scene', 'graph') else 'time_bits'
        if all(isinstance(n, dict) and type(n.get(identity)) is int for n in a + b):
            left, right = ({str(n[identity]): n for n in rows} for rows in (a, b))
            if len(left) == len(a) and len(right) == len(b):
                return difference(left, right, path) or dict(path=path, kind='order',
                    expected=digest([n[identity] for n in a]), actual=digest([n[identity] for n in b]))
        if len(a) != len(b):
            return dict(path=path, kind='length', expected=digest(a), actual=digest(b))
        for index, (x, y) in enumerate(zip(a, b)):
            found = difference(x, y, path + (str(index),))
            if found:
                return found
    return dict(path=path, kind='value', type=type(a).__name__, expected=a, actual=b)


def mismatch(cases, wanted, actual):
    for index, (case, a, b) in enumerate(zip(cases, wanted, actual)):
        found = difference(a, b)
        if found:
            return dict(index=index, expected=a, actual=b, signature=dict(origin=case.origin,
                expected_result=a['result'], actual_result=b.get('result') if isinstance(b, dict) else None,
                difference=found))
    return None


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate native JSON key')
        value[key] = item
    return value


def reject_constant(value):
    raise ValueError('nonfinite native JSON constant: ' + value)


def validate_record(row):
    """Validate report framing/types, leaving scene semantics to the oracle.

    Output bytes are bounded by process supervision. Do not turn syntactically
    valid but structurally invalid reports into resolved semantic comparisons.
    """
    def require(condition):
        if not condition:
            raise ValueError('invalid native record schema')

    def keys(value, names):
        require(type(value) is dict and value.keys() == set(names))

    def strings(value):
        return type(value) is list and all(type(item) is str for item in value)

    def unsigned(value, bits):
        return type(value) is int and 0 <= value < 2**bits

    def field(value):
        keys(value, ('type', 'value'))
        kind, raw = value['type'], value['value']
        require(type(kind) is str)
        if kind == 'path_op':
            buckets = ('explicit_items', 'added', 'prepended', 'appended', 'deleted', 'ordered')
            keys(raw, ('explicit', *buckets))
            require(type(raw['explicit']) is bool and all(strings(raw[b]) for b in buckets))
        elif kind == 'time_samples':
            require(type(raw) is list)
            previous = None
            for item in raw:
                keys(item, ('time_bits', 'value'))
                require(unsigned(item['time_bits'], 64))
                time = struct.unpack('<d', struct.pack('<Q', item['time_bits']))[0]
                require(not math.isnan(time) and (previous is None or previous < time))
                previous = time
                require(type(item['value']) is dict and item['value'].get('type') != 'time_samples')
                field(item['value'])
        elif kind in ('tokens', 'paths', 'string[]', 'token[]'):
            require(strings(raw))
        elif kind == 'string':
            require(type(raw) is str)
        elif kind.removesuffix('[]') in REAL_FAMILIES and kind != 'double':
            _, fmt, width = REAL_FAMILIES[kind.removesuffix('[]')]
            require(type(raw) is list and all(unsigned(bits, 32 if fmt == 'f' else 64) for bits in raw))
            require(len(raw) % width == 0 if kind.endswith('[]') else len(raw) == width)
        elif kind == 'double':
            require(type(raw) is float and math.isfinite(raw))
        elif kind == 'int32':
            require(type(raw) is int and -(2**31) <= raw < 2**31)
        elif kind in ('specifier', 'variability'):
            require(unsigned(raw, 32))
        elif kind == 'bool':
            require(type(raw) is bool)
        elif kind == 'token':
            require(type(raw) is str)
        else:
            require(False)

    keys(row, ('result', 'scene'))
    require(type(row['result']) is str and row['result'] in ('ok', 'unsupported', 'malformed', 'limit'))
    scene = row['scene']
    keys(scene, ('source_format', 'graph'))
    require(type(scene['source_format']) is str and type(scene['graph']) is list)
    identities = set()
    for node in scene['graph']:
        keys(node, ('id', 'parent', 'type', 'name', 'order', 'path', 'fields'))
        require(unsigned(node['id'], 64) and node['id'] not in identities)
        identities.add(node['id'])
        require(unsigned(node['parent'], 64) and unsigned(node['type'], 32) and unsigned(node['order'], 32))
        require(type(node['name']) is str and type(node['path']) is str and type(node['fields']) is dict)
        for name, value in node['fields'].items():
            require(type(name) is str)
            field(value)


class Trials:
    """Fresh process and independent model for each candidate, with durable plans."""
    def __init__(self, root, command, cwd, environment, *, timeout_ms=1000, runner=run_process):
        self.root, self.command, self.cwd = Path(root), list(command), cwd
        self.environment, self.timeout_ms, self.runner = environment, timeout_ms, runner
        self.count = 0
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / 'blobs').mkdir()
        self.bootstrap = self.root / 'bootstrap.ompf'
        self.bootstrap.write_bytes(encode(Delta()))

    def store(self, cases):
        records = []
        for case in cases:
            record = case.reference()
            path = self.root / 'blobs' / (record['sha256'] + '.ompf')
            if path.exists():
                if path.read_bytes() != case.payload:
                    raise AssertionError('reduction blob identity mismatch')
            else:
                path.write_bytes(case.payload)
            records.append(record)
        return records

    def evaluate(self, cases):
        self.count += 1
        directory = self.root / f'trial-{self.count:04d}'
        directory.mkdir()
        refs = self.store(cases)
        command = self.command + [str(self.bootstrap)] + [str(self.root / 'blobs' / (r['sha256'] + '.ompf')) for r in refs]
        report = dict(cases=refs, command=command, outcome='pending')
        write_json(directory / 'result.json', report)
        wanted = expected(cases)
        result = self.runner(command, cwd=self.cwd, output=directory / 'native.log', timeout_ms=self.timeout_ms,
                             max_output_bytes=8*1024*1024, environment=self.environment)
        report['process'] = asdict(result)
        report['outcome'] = 'unresolved'
        if result.outcome == 'passed' and result.cleanup_complete and not result.output_truncated:
            try:
                rows = [json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
                        for line in (directory / 'native.log').read_text().splitlines()]
                if len(rows) != len(cases) + 1 or canonical(rows[0]) != canonical(expected([Case(-1, 'bootstrap', self.bootstrap.read_bytes())])[0]):
                    raise ValueError('missing, extra or invalid bootstrap/native rows')
                for row in rows[1:]:
                    validate_record(row)
                report['mismatch'] = mismatch(cases, wanted, rows[1:])
                report['outcome'] = 'mismatch' if report['mismatch'] else 'passed'
            except (ValueError, TypeError, RecursionError):
                report['reason'] = 'invalid native report'
        write_json(directory / 'result.json', report)
        return report


def reduce_history(cases, target, trials, *, max_trials=48, context=None):
    """Delete batches, commands, fields and samples under an exact failure signature.

    No global minimality or OS scheduling claim. A final fresh execution is
    reserved even when the candidate budget is exhausted.
    """
    if type(max_trials) is not int or max_trials < 2 or max_trials > 256 or not cases or len(cases) > 1023:
        raise ValueError('bounded nonempty history and 2..256 trials required')
    if (any(not isinstance(c, Case) or type(c.origin) is not int or type(c.name) is not str or
            type(c.payload) is not bytes for c in cases) or
            len({c.origin for c in cases}) != len(cases) or
            type(target.get('origin')) is not int or target['origin'] not in {c.origin for c in cases}):
        raise ValueError('unique immutable case identities and existing target required')
    if any(len(c.payload) > 1024*1024 for c in cases) or sum(len(c.payload) for c in cases) > 64*1024*1024:
        raise ValueError('reduction payload byte limit')
    if trials.count:
        raise ValueError('reduction requires fresh trials')
    best = tuple(cases)
    cache, unresolved = {}, False
    status = dict(schema_version=1, status='running', context=context or {}, target=target, original=trials.store(cases),
                  cases=trials.store(best), attempts=0, verified=False, deletion_minimal=False,
                  scope='batch/command/field/sample deletion under fixed first-mismatch signature')

    def save():
        status.update(cases=trials.store(best), attempts=trials.count)
        write_json(trials.root / 'reduction.json', status)

    def matching(report):
        return report['outcome'] == 'mismatch' and canonical(report['mismatch']['signature']) == canonical(target)

    save()
    first = trials.evaluate(best)
    if not matching(first):
        status['status'] = 'not_reproduced'; save(); return status

    class Budget(Exception):
        pass

    def accept(candidate):
        nonlocal best, unresolved
        if not candidate or all(c.origin != target['origin'] for c in candidate):
            return False
        key = tuple((c.origin, hashlib.sha256(c.payload).digest()) for c in candidate)
        if key in cache:
            if cache[key]:
                best = tuple(candidate)
                save()
            return cache[key]
        if trials.count >= max_trials - 1:
            raise Budget()
        report = trials.evaluate(candidate)
        unresolved |= report['outcome'] == 'unresolved'
        same = cache[key] = matching(report)
        if same:
            best = tuple(candidate)
            save()
        return same

    def remove_chunks(items, candidate):
        items, divisions = tuple(items), 2
        while items:
            width = max(1, (len(items) + divisions - 1) // divisions)
            for begin in range(0, len(items), width):
                remaining = items[:begin] + items[begin + width:]
                if accept(candidate(remaining)):
                    items, divisions = remaining, max(2, divisions - 1)
                    break
            else:
                if divisions >= len(items):
                    return
                divisions = min(len(items), divisions * 2)

    skipped = False
    try:
        while True:
            before = best
            remove_chunks(best, lambda remaining: remaining)
            for case in tuple(best):
                try:
                    delta = decode(case.payload)
                except GraphWireError:
                    skipped = True
                    continue  # Preserve raw malformed input; no guessed reconstruction.
                for field in ('commands', 'fields', 'samples'):
                    def candidate(remaining, origin=case.origin, delta=delta, field=field):
                        payload = encode(replace(delta, **{field: remaining}))
                        return tuple(replace(c, payload=payload) if c.origin == origin else c for c in best)
                    remove_chunks(getattr(delta, field), candidate)
                    delta = decode(next(c.payload for c in best if c.origin == case.origin))
            if best == before:
                status['status'] = 'complete'
                break
    except Budget:
        status['status'] = 'budget_exhausted'
    final = trials.evaluate(best)
    status.update(verified=matching(final), unresolved_trials=unresolved,
                  record_deletion_skipped=skipped, final_trial=trials.count)
    if not status['verified']:
        status['status'] = 'unstable'
    status['deletion_minimal'] = status['status'] == 'complete' and not unresolved and not skipped
    save()
    return status
