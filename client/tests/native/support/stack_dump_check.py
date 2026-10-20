# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks that a native test driver answers SIGQUIT with every thread's stack.

Usage: stack_dump_check.py <live_stack_dump_probe> <none|address|thread>. The
probe parks two threads in liveStackDumpDeadlockOnPurpose and a third that
blocks every signal, then prints ready; this sends SIGQUIT, waits for the
dump's end line in <prefix>.<pid>, then does so again. Each of the two dumps
lists the parked threads and counts only its own unanswered threads as missing.
"""
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in {"none", "address", "thread"}:
        print(__doc__, file=sys.stderr)
        return 2
    probe, kind = sys.argv[1], sys.argv[2]
    with tempfile.TemporaryDirectory() as directory:
        prefix = Path(directory) / "probe.stacks"
        child = subprocess.Popen([probe], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                 env=dict(os.environ, LIVE_STACK_DUMP_PREFIX=str(prefix)))
        try:
            ready = child.stdout.readline()
            if ready != b"ready\n":
                print(f"live stack dump: probe did not become ready: {ready!r}", file=sys.stderr)
                return 1
            dump = Path(f"{prefix}.{child.pid}")
            for count in (1, 2):
                child.send_signal(signal.SIGQUIT)
                deadline = time.monotonic() + 10  # Failure-only watchdog.
                while not (dump.is_file() and dump.read_text(errors="replace").count("live-stack-dump end") == count):
                    if child.poll() is not None or time.monotonic() >= deadline:
                        print(f"live stack dump: no complete dump {count} (probe exit {child.poll()})", file=sys.stderr)
                        return 1
                    time.sleep(0.01)
            text = dump.read_text(errors="replace")
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()
            child.stdout.close()
    # The blocked thread never answers; ThreadSanitizer's background thread runs no user handler.
    allowed = range(1, 3 if kind == "thread" else 2)
    markers = re.findall(r"^live-stack-dump (begin|end) pid=\d+(?: missing=(-?\d+))?$", text, re.M)
    dumps = text.split("live-stack-dump begin ")[1:]
    passed = [kind for kind, _ in markers] == ["begin", "end", "begin", "end"] and len(dumps) == 2
    for dump in dumps:
        blocks = dump.split("live-stack-thread ")[1:]
        deadlocked = sum("liveStackDumpDeadlockOnPurpose" in block for block in blocks)
        missing = re.search(r"^live-stack-dump end pid=\d+ missing=(-?\d+)$", dump, re.M)
        missing = int(missing.group(1)) if missing else None
        print(f"live stack dump: {len(blocks)} threads, missing {missing}")
        passed = passed and len(blocks) >= 2 and deadlocked >= 2 and missing in allowed
    if not passed:
        print(text, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
