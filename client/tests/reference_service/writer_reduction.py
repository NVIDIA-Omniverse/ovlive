# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded operation deletion using fresh public-plugin/reference-service trials.

The evaluator must return only after all its processes and service have retired.
Only a matching semantic signature can retain a deletion; infrastructure errors,
timeouts and unrelated failures never do. OS scheduling is not replayed.
"""
import copy
import hashlib
import json

from writer_history import validate


def publish(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()


async def reduce_history(plan, target, root, evaluate, *, max_trials=16):
    """Delete chunks, then individual edits, keeping every original writer active.

    evaluate(plan, directory) returns {outcome: passed|mismatch|unresolved,
    signature?: {...}, cleanup_complete: bool}. Original and every proposed plan
    are saved before evaluation, including the pending attempt on interruption.
    """
    validate(plan)
    if type(max_trials) is not int or not 2 <= max_trials <= 64:
        raise ValueError('writer reduction requires 2..64 trials')
    root.mkdir(parents=True, exist_ok=False)
    original = copy.deepcopy(plan)
    best = copy.deepcopy(plan)
    publish(root / 'original.plan.json', original)
    summary = dict(schema_version=1, status='running', original_sha256=digest(original),
                   target=target, attempts=0, unresolved_trials=0, verified=False,
                   operation_deletion_minimal=False, replay_scope=plan['replay_scope'],
                   domain='operation deletion with original nonempty writers and first-RPC hold recipe')

    def checkpoint():
        publish(root / 'best.plan.json', best)
        summary['best_sha256'] = digest(best)
        summary['retained_edits'] = [op['edit'] for writer in best['writers'] for op in writer]
        publish(root / 'reduction.json', summary)

    async def trial(candidate):
        summary['attempts'] += 1
        directory = root / f"trial-{summary['attempts']:04d}"
        directory.mkdir()
        publish(directory / 'input.plan.json', candidate)
        publish(directory / 'result.json', dict(outcome='pending', plan_sha256=digest(candidate)))
        checkpoint()
        result = await evaluate(copy.deepcopy(candidate), directory)
        # Fail closed even if the evaluator produced a matching signature before
        # a shutdown error. Cancellation propagates and leaves this trial's result.json pending.
        if (not isinstance(result, dict) or result.get('cleanup_complete') is not True or
                result.get('outcome') not in ('passed', 'mismatch') or
                (result.get('outcome') == 'mismatch' and not isinstance(result.get('signature'), dict))):
            result = dict(outcome='unresolved', cleanup_complete=False, evidence=result)
        publish(directory / 'result.json', dict(result, plan_sha256=digest(candidate)))
        if result['outcome'] == 'unresolved':
            summary['unresolved_trials'] += 1
        checkpoint()
        return result['outcome'] == 'mismatch' and result['signature'] == target

    checkpoint()
    if not await trial(best):
        summary['status'] = 'not_reproduced'
        checkpoint()
        return summary
    tested = {digest(best)}
    exhausted = False
    # A successful deletion restarts the pass: outcomes can be nonmonotonic.
    while True:
        edits = [op['edit'] for writer in best['writers'] for op in writer]
        changed = False
        width = max(1, len(edits) // 2)
        while width:
            for offset in range(0, len(edits), width):
                removed = set(edits[offset:offset + width])
                candidate = copy.deepcopy(best)
                candidate['writers'] = [[op for op in writer if op['edit'] not in removed]
                                        for writer in candidate['writers']]
                if any(not writer for writer in candidate['writers']):
                    continue
                identity = digest(candidate)
                if identity in tested:
                    continue
                if summary['attempts'] >= max_trials - 1:  # Reserve fresh final confirmation.
                    exhausted = True
                    break
                tested.add(identity)
                if await trial(candidate):
                    best = candidate
                    # A failed candidate against an earlier larger history says
                    # nothing about deletion minimality of this smaller history.
                    tested = {digest(best)}
                    checkpoint()
                    changed = True
                    break
            if changed or exhausted:
                break
            width = width // 2 if width > 1 else 0
        if exhausted or not changed:
            break
    summary['verified'] = await trial(best)
    summary['status'] = ('unstable' if not summary['verified'] else
                         'budget_exhausted' if exhausted else 'complete')
    summary['operation_deletion_minimal'] = (summary['status'] == 'complete' and
                                              summary['unresolved_trials'] == 0)
    checkpoint()
    return summary
