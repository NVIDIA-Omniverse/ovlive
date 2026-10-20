# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline check of uncertain-writer exports; standard library only, no service.

Each writer exports its failed live layer with SdfLayer::Export. After every
original process has exited, this reads the plan, the recorded terminal state,
the service ledger and the exported USDA files. Each export must be the
writer's frozen host layer: every local edit over a prefix of the service
ledger, a prefix no shorter than the one the writer showed before the cut.
The ledger must hold only first writes and the harness peer writes: no queued
write issued, no uncertain write replayed, no wire writer committing twice.
The plan and ledger are harness evidence, never a client retry input. The
bounded profile is root-int32 authoring; any other export line is rejected.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

from scene_wire import decode_update

ROOT_FIELD = re.compile(r'\s*(?:int\s+)?((?:owned|shared)_[A-Za-z_0-9]+)\s*=\s*(-?\d+)\s*')


class EvidenceError(AssertionError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def exported_fields(text):
    require(text.startswith('#usda 1.0'), 'export is not a USDA layer')
    lines = text.splitlines()
    require(lines[0] == '#usda 1.0', 'export is not a USDA layer')
    body = [line.strip() for line in lines[1:] if line.strip()]
    if body:
        require(body[0] == '(' and body[-1] == ')', 'export body is not one root metadata block')
        body = body[1:-1]
    fields = {}
    for line in body:
        match = ROOT_FIELD.fullmatch(line)
        require(match is not None, 'export holds a line outside the root-int32 profile')
        name, value = match.groups()
        require(name not in fields, 'duplicate exported root field')
        fields[name] = int(value)
    return fields


def verify(text, expected, operations, commits):
    exported = exported_fields(text)
    state = expected['state']
    require((state['state'], state['result'], state['pending']) == (7, 19, len(operations)), 'terminal status')
    status = expected['status']
    require(status['state'] == 7 and int(status['subject']) > 0 and status['incident'] > 0, 'terminal provenance')
    require(exported == state['fields'], 'export differs from the frozen host layer')
    local = {}
    for op in operations:
        local[op['field']] = op['value']
    require(all(exported.get(field) == value for field, value in local.items()), 'export lost a local edit')
    # The host freezes at its last installed revision; local edits overlay
    # every installed peer write, and installation never moves backwards.
    installed, frozen = {}, [dict(local)]
    for commit in sorted(commits, key=lambda entry: entry['sequence']):
        installed.update(decode_update(bytes.fromhex(commit['wire'])))
        frozen.append(dict(installed, **local))
    require(expected['before_cut'] in frozen, 'host layer lost a local edit before the cut')
    floor = frozen.index(expected['before_cut'])
    require(exported in frozen, 'export holds an unapplied peer value or lost an installed one')
    require(exported in frozen[floor:], 'export is below the ledger prefix shown before the cut')
    return dict(subject=status['subject'], incident=status['incident'], operations=len(operations),
                fields=len(exported))


def verify_ledger(plan, commits, peers):
    # Every permitted write has one fingerprint, so a matching commit names
    # exactly one planned first write or harness peer write.
    firsts = Counter(json.dumps({ops[0]['field']: ops[0]['value']}, sort_keys=True)
                     for ops in plan['writers'] if ops)
    firsts.update(json.dumps(peer, sort_keys=True) for peer in peers)
    require(all(count == 1 for count in firsts.values()), 'permitted write fingerprints are not unique')
    issued = Counter(json.dumps(decode_update(bytes.fromhex(commit['wire'])), sort_keys=True) for commit in commits)
    extra = issued - firsts
    require(not any(key not in firsts for key in extra), 'a queued write was issued')
    require(not extra, 'an uncertain write was replayed')
    writers = Counter(commit['writer'] for commit in commits)
    require(all(count == 1 for count in writers.values()), 'a wire writer committed more than once')


def main(path):
    evidence = json.loads(Path(path).read_text())
    reports = []
    for index, expected in enumerate(evidence['clients']):
        data = Path(expected['export']).read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected['sha256'], 'export changed after producer exit')
        reports.append(verify(data.decode(), expected, evidence['plan']['writers'][index], evidence['commits']))
    require('peers' in evidence, 'evidence omits the harness peer writes')
    verify_ledger(evidence['plan'], evidence['commits'], evidence['peers'])
    print(json.dumps(dict(schema_version=2, clients=reports,
                          verified_operations=sum(report['operations'] for report in reports)), sort_keys=True))


if __name__ == '__main__':
    main(sys.argv[1])
