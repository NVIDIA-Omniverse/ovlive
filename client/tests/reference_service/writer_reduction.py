# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded history deletion using fresh client/reference-service trials.

A reduction domain names the units a plan's history is deleted by: the root-field
domain deletes single edits; the round domain deletes whole writers, rounds and
backend rotations of a structural or sample plan; the period domain first cuts a
long-session plan after its failing period, then deletes whole cycles and writers; the staged-samples domain deletes
whole participant samples with the writers on their layers, writers and rounds. The evaluator must return only
after all its processes and service have retired. Only a matching semantic
signature can retain a deletion; infrastructure errors, timeouts and unrelated
failures never do. OS scheduling is not replayed.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Callable

import period_plans
import round_plans
import staged_sample_history
import writer_history


def publish(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Domain:
    """What a reduction deletes: `units(plan)` lists a plan's deletable units in order, `remove(plan, units)`
    returns the plan without them or None when no trial could execute it, and `retained(plan)` names its edits.

    `truncate(plan, target)`, when a domain has it, returns the plan cut after the failing step the target signature
    names, which executes exactly the plan's inputs through that step, or None; the reducer's first trial then
    replays the cut plan in place of the whole one.
    """
    description: str
    validate: Callable
    units: Callable
    remove: Callable
    retained: Callable
    truncate: Callable = None


def _edits(plan):
    return [op['edit'] for writer in plan['writers'] for op in writer]


def _remove_edits(plan, removed):
    candidate = copy.deepcopy(plan)
    candidate['writers'] = [[op for op in writer if op['edit'] not in removed] for writer in candidate['writers']]
    return None if any(not writer for writer in candidate['writers']) else candidate


ROOT_FIELDS = Domain('operation deletion with original nonempty writers and first-RPC hold recipe',
                     writer_history.validate, _edits, _remove_edits, _edits)


def round_domain(validate, origin):
    """Whole-unit deletion for a round-based plan (round_plans): every retained writer authors one edit in every
    retained round.

    `validate(plan)` accepts an executable plan and raises ValueError otherwise; `origin(plan)` regenerates the
    seeded plan `plan` was selected from. Units are `rotation:<ordinal>`, `writer:<index>` and `round:<ordinal>`, by
    original index and ordinal; deleting a round deletes the rotation after it. A candidate is the seeded origin's
    selection, so it carries no edit the generator did not author.
    """
    def selection(plan):
        writers = [writer[0]['writer'] for writer in plan['writers']]
        ordinals = [op['ordinal'] for op in plan['writers'][0]]
        return writers, ordinals, [ordinals[position] for position in plan['rotation_after']]

    def units(plan):
        writers, ordinals, rotations = selection(plan)
        return ([f'rotation:{o}' for o in rotations] + [f'writer:{w}' for w in writers] +
                [f'round:{o}' for o in ordinals])

    def remove(plan, removed):
        writers, ordinals, rotations = selection(plan)
        writers = [w for w in writers if f'writer:{w}' not in removed]
        ordinals = [o for o in ordinals if f'round:{o}' not in removed]
        rotations = [o for o in rotations if f'rotation:{o}' not in removed and o in ordinals]
        if not writers or not ordinals:
            return None
        candidate = round_plans.select(origin(plan), writers, ordinals, rotations)
        if 'test_service_fault' in plan:
            candidate['test_service_fault'] = plan['test_service_fault']
        try:
            return validate(candidate)
        except ValueError:
            return None  # A retained edit depends on a deleted one, such as a field edit on an uncreated branch.

    return Domain('writer, round and rotation deletion from the seeded origin with per-round barriers',
                  validate, units, remove, _edits)


def period_domain():
    """Truncation after the failing period, then whole-cycle and writer deletion, for a period plan (period_plans).

    The first trial replays the plan cut after the seeded `period` the target signature names, so it and every later
    trial run only the failure's prefix. Units are `cycle:<c>` (periods 2c + 1 and 2c + 2) and `writer:<index>`, by
    seeded index; the last retained cycle holds the failing period once the plan is cut, so no candidate deletes it.
    """
    def selection(plan):
        return [writer[0]['writer'] for writer in plan['writers']], [op['ordinal'] for op in plan['writers'][0]]

    def cycles(plan):
        return list(dict.fromkeys(ordinal // 2 for ordinal in selection(plan)[1]))

    def units(plan):
        return [f'cycle:{c}' for c in cycles(plan)] + [f'writer:{w}' for w in selection(plan)[0]]

    def remove(plan, removed):
        writers, ordinals = selection(plan)
        if f'cycle:{cycles(plan)[-1]}' in removed:
            return None
        writers = [w for w in writers if f'writer:{w}' not in removed]
        if not writers:
            return None
        return period_plans.reselect(plan, writers, [o for o in ordinals if f'cycle:{o // 2}' not in removed])

    def truncate(plan, target):
        return period_plans.truncate(plan, target.get('period'))

    return Domain('truncation after the failing period, then cycle and writer deletion from the seeded origin',
                  period_plans.validate, units, remove, _edits, truncate)


def staged_domain():
    """Whole-unit deletion for a staged-samples plan (staged_sample_history): every retained participant authors one
    edit in every retained round.

    Units are `sample:<name>`, `writer:<index>` and `round:<ordinal>`, by seeded name, index and ordinal; deleting a
    sample deletes its layer and so the writers on it, and every candidate keeps a sample and a round. A candidate is
    the seeded origin's selection, so it carries no edit the generator did not author, and none in which a writer sets
    its prim without the round that defines it.
    """
    def units(plan):
        samples, writers, ordinals = staged_sample_history.selection(plan)
        return ([f'sample:{name}' for name in samples] + [f'writer:{w}' for w in writers] +
                [f'round:{o}' for o in ordinals])

    def remove(plan, removed):
        samples, writers, ordinals = staged_sample_history.selection(plan)
        samples = [name for name in samples if f'sample:{name}' not in removed]
        layers = dict(zip(writers, plan['layers']))
        writers = [w for w in writers if f'writer:{w}' not in removed and layers[w] in samples]
        ordinals = [o for o in ordinals if f'round:{o}' not in removed]
        if not samples or not ordinals:
            return None
        try:
            return staged_sample_history.reselect(plan, samples, writers, ordinals)
        except ValueError:
            return None  # A retained writer sets its prim without the round that defines it.

    return Domain('sample, writer and round deletion from the seeded origin with per-round barriers',
                  staged_sample_history.validate, units, remove, staged_sample_history.edits)


async def reduce_history(plan, target, root, evaluate, *, max_trials=16, domain=ROOT_FIELDS, context=None):
    """Delete chunks, then individual units of `domain`, keeping every candidate executable.

    evaluate(plan, directory) returns {outcome: passed|mismatch|unresolved,
    signature?: {...}, cleanup_complete: bool}. Original and every proposed plan
    are saved before evaluation, including the pending attempt on interruption.
    A domain that truncates replays the plan cut after the failing step as the first
    trial, never the whole plan, and the summary's `truncated` says so.
    `reduction.json` is schema version 2, whose summary names its retained edits
    and the domain's retained units and reports `deletion_minimal`; a `context`,
    when given, is recorded as the summary's replay context (plan_replay).
    """
    domain.validate(plan)
    if type(max_trials) is not int or not 2 <= max_trials <= 64:
        raise ValueError('writer reduction requires 2..64 trials')
    root.mkdir(parents=True, exist_ok=False)
    original = copy.deepcopy(plan)
    best = copy.deepcopy(plan)
    publish(root / 'original.plan.json', original)
    summary = dict(schema_version=2, status='running', original_sha256=digest(original),
                   target=target, attempts=0, unresolved_trials=0, verified=False,
                   deletion_minimal=False, replay_scope=plan['replay_scope'], domain=domain.description)
    if context is not None:
        summary['context'] = context

    def checkpoint():
        publish(root / 'best.plan.json', best)
        summary['best_sha256'] = digest(best)
        summary['retained_edits'] = domain.retained(best)
        summary['retained_units'] = domain.units(best)
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
    # The cut plan runs exactly the original's inputs through the failing step, so it alone decides reproduction.
    truncated = domain.truncate(best, target) if domain.truncate else None
    summary['truncated'] = truncated is not None
    if not await trial(best if truncated is None else truncated):
        summary['status'] = 'not_reproduced'
        checkpoint()
        return summary
    if truncated is not None:
        best = truncated
        checkpoint()
    tested = {digest(best)}
    exhausted = False
    # A successful deletion restarts the pass: outcomes can be nonmonotonic.
    while True:
        units = domain.units(best)
        changed = False
        width = max(1, len(units) // 2)
        while width:
            for offset in range(0, len(units), width):
                candidate = domain.remove(best, set(units[offset:offset + width]))
                if candidate is None:
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
    summary['deletion_minimal'] = summary['status'] == 'complete' and summary['unresolved_trials'] == 0
    checkpoint()
    return summary
