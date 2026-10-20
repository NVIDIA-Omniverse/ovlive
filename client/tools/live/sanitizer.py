# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The runtime recipes for instrumented native lanes.

`address` is ASan, LSan and UBSan with leak detection on; `lsan.supp` names
third-party frames only, each with an owner and reason. `thread` is TSan with
`tsan.supp`, whose entries key on a third-party frame in the report's stacks:
an uninstrumented third-party library or a third-party header template inlined
into our code, never a product frame. Runs apply
`runtime_options()`; recorded evidence uses `provenance()`, which names each
suppressions file by content hash and the TSan log path by a fixed token so
replays stay portable across checkouts and run directories.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SUPPRESSIONS = ROOT / 'tools/live/lsan.supp'
SUPPRESSIONS_TSAN = ROOT / 'tools/live/tsan.supp'
PASSTHROUGH = ('ASAN_OPTIONS', 'LSAN_OPTIONS', 'UBSAN_OPTIONS', 'TSAN_OPTIONS')
# The personality flag `setarch -R` sets; GCC 12's libtsan needs it on high-entropy mmap kernels.
ADDR_NO_RANDOMIZE = 0x0040000


def kind_of(identity) -> str:
    """The sanitizer kind a native identity records; `sanitizers` alone names the address build."""
    return identity.get('sanitizer', 'address' if identity.get('sanitizers') else 'none')


def option_value(path) -> str:
    """A path as a sanitizer option value: double-quoted when it holds the option separator `:`, as a Windows drive
    does, which the sanitizer runtimes' flag parser accepts."""
    text = str(path)
    return f'"{text}"' if ':' in text else text


def split_options(text) -> list:
    """The `:`-separated options of a sanitizer option string; a double-quoted value may hold `:`."""
    return re.findall(r'(?:[^:"]|"[^"]*")+', text)


def runtime_options(kind='address', log_prefix=None) -> dict:
    if kind == 'thread':
        if log_prefix is None:
            raise ValueError('The thread recipe needs a log prefix')
        # history_size=7 keeps the previous stack of the completion-queue handoff reports.
        # halt_on_error=0 with exitcode=66 logs every report and still fails the process.
        # log_path keeps complete per-pid reports that a suite's bounded output tail would truncate.
        return {'TSAN_OPTIONS': f'suppressions={option_value(SUPPRESSIONS_TSAN)}:history_size=7:'
                                f'second_deadlock_stack=1:detect_deadlocks=1:halt_on_error=0:exitcode=66:'
                                f'log_path={option_value(log_prefix)}'}
    if kind != 'address':
        raise ValueError('Unknown sanitizer kind: ' + str(kind))
    # The USD SDK is built without frame pointers, so the fast unwinder
    # truncates allocation stacks before the suppressed USD frames. Full
    # unwinding costs allocation speed in the sanitizer lanes only.
    # The sanitized samples must keep stderr empty for the package check, so
    # LSan does not print matched suppressions; an unsuppressed leak still
    # fails the lane.
    return {'ASAN_OPTIONS': 'detect_leaks=1:fast_unwind_on_malloc=0:halt_on_error=1',
            'LSAN_OPTIONS': f'suppressions={option_value(SUPPRESSIONS)}:print_suppressions=0',
            'UBSAN_OPTIONS': 'halt_on_error=1:print_stacktrace=1'}


def recorded(environment) -> dict:
    """The portable record of the sanitizer options present in `environment`."""
    result = {key: environment[key] for key in PASSTHROUGH if key in environment}
    for key in ('LSAN_OPTIONS', 'TSAN_OPTIONS'):
        if key not in result:
            continue
        options = []
        for option in split_options(result[key]):
            if option.startswith('suppressions='):
                path = option[len('suppressions='):].strip('"')
                digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                option = 'suppressions=sha256:' + digest
            elif option.startswith('log_path='):
                option = 'log_path=run-directory'
            options.append(option)
        result[key] = ':'.join(options)
    return result


def provenance(kind='address') -> dict:
    return recorded(runtime_options(kind, ROOT))


def address_randomization_disabled() -> bool:
    """Whether this process runs with ASLR disabled, as under `setarch -R`."""
    try:
        return bool(int(Path('/proc/self/personality').read_text().strip(), 16) & ADDR_NO_RANDOMIZE)
    except (OSError, ValueError):
        return False
