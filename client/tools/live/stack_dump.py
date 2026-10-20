# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Python thread stacks on SIGQUIT for test processes (docs/ovlive-testing.md)."""
import faulthandler
import os
import signal

_file = None


def install():
    """Registers faulthandler for SIGQUIT when LIVE_STACK_DUMP_PREFIX is set; otherwise does nothing."""
    global _file
    prefix = os.environ.get('LIVE_STACK_DUMP_PREFIX')
    if not prefix or _file is not None or not hasattr(signal, 'SIGQUIT'):
        return
    # faulthandler writes to this descriptor from its C handler for the process lifetime.
    _file = open(f'{prefix}.{os.getpid()}', 'w')
    faulthandler.register(signal.SIGQUIT, file=_file, all_threads=True, chain=False)
