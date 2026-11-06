# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks that a native test driver answers SIGQUIT with every thread's stack.

Usage: stack_dump_check.py <live_stack_dump_probe> <none|address|thread>. The
probe parks two threads in liveStackDumpDeadlockOnPurpose and a third that
blocks every signal, then prints ready; this sends SIGQUIT, waits for the
dump's end line in <prefix>.<pid>, then does so again. Each of the two dumps
lists the parked threads and counts only its own unanswered threads as missing.
On x86-64 and aarch64 the probe parks a fourth thread below a frame whose unwind
information is wrong on purpose; each dump lists it with the frames reached
before the unwinder faulted and a live-stack-unwind-fault line, and the probe
survives both dumps.

Then it runs each case of live_stack_dump_cases, built beside the probe
(stack_dump_cases.cpp says what each sets up), and checks its dumps: the
raising thread counts as missing when it gets no guard, a late answer counts
toward no dump but the open one, and a dump started on a thread still answering
an ended dump contains its fault and ends.
"""
import os
from pathlib import Path
import platform
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
            if child.poll() is not None:
                print(f"live stack dump: the probe ended after its dumps (exit {child.poll()})", file=sys.stderr)
                return 1
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()
            child.stdout.close()
    # The blocked thread never answers; ThreadSanitizer's background thread runs no user handler.
    extra = range(0, 2 if kind == "thread" else 1)
    bad_unwind = platform.machine().lower() in {"x86_64", "amd64", "aarch64", "arm64"}
    dumps = split(text, 2)
    passed = dumps is not None
    for blocks, missing in dumps or []:
        deadlocked = sum("liveStackDumpDeadlockOnPurpose" in block for block in blocks)
        faulted = [block for block in blocks if FAULT.search(block)]
        print(f"live stack dump: {len(blocks)} threads, {len(faulted)} unwind faults, missing {missing}")
        passed = passed and len(blocks) >= 2 and deadlocked >= 2 and missing - 1 in extra
        if bad_unwind:
            passed = passed and len(faulted) == 1 and "liveStackDumpBadUnwindFrame" in faulted[0]
        else:
            passed = passed and not faulted
    if not passed:
        print(text, file=sys.stderr)
        return 1
    cases = Path(probe).with_name("live_stack_dump_cases" + Path(probe).suffix)
    for case, check in CASES.items():
        text = run_case(cases, case)
        dumps = split(text, check["dumps"]) if text is not None else None
        if dumps is None or not check["check"](dumps, extra, kind):
            print(f"live stack dump: case {case} failed", file=sys.stderr)
            if text is not None:
                print(text, file=sys.stderr)
            return 1
        print(f"live stack dump: case {case}: " + ", ".join(
            f"{len(blocks)} threads, missing {missing}" for blocks, missing in dumps))
    return 0


FAULT = re.compile(r"^live-stack-unwind-fault signal=\d+$", re.M)


def split(text, count):
    """Each dump's thread blocks and missing count, or None unless there are `count` complete dumps."""
    markers = re.findall(r"^live-stack-dump (begin|end) pid=\d+(?: missing=(-?\d+))?$", text, re.M)
    if [marker for marker, _ in markers] != ["begin", "end"] * count:
        return None
    dumps = []
    for dump in text.split("live-stack-dump begin ")[1:]:
        missing = re.search(r"^live-stack-dump end pid=\d+ missing=(-?\d+)$", dump, re.M)
        dumps.append((dump.partition("live-stack-dump end ")[0].split("live-stack-thread ")[1:], int(missing.group(1))))
    return dumps


def named(blocks, name):
    return [block for block in blocks if re.match(rf"tid=\d+ name={name}\n", block)]


def run_case(cases, case):
    """The dump text of one live_stack_dump_cases run, or None when it did not print done and exit 0."""
    with tempfile.TemporaryDirectory() as directory:
        prefix = Path(directory) / "case.stacks"
        try:
            result = subprocess.run([str(cases), case], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, timeout=60,
                                    env=dict(os.environ, LIVE_STACK_DUMP_PREFIX=str(prefix)))  # Failure-only watchdog.
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"live stack dump: case {case}: {error}", file=sys.stderr)
            return None
        text = "".join(path.read_text(errors="replace") for path in Path(directory).glob("case.stacks.*"))
        if result.returncode != 0 or result.stdout != b"done\n":
            print(f"live stack dump: case {case} exit {result.returncode}, output {result.stdout!r}", file=sys.stderr)
            print(text, file=sys.stderr)
            return None
        return text


def full_guards(dumps, extra, kind):
    # The raising thread got no guard: no block, and the end line counts it.
    (blocks, missing), = dumps
    return not blocks and missing - 1 in extra


def late_answer(dumps, extra, kind):
    # The second dump has one block from the thread that answered both dumps at once and still counts the other.
    # ThreadSanitizer keeps one pending instance of a signal, the first dump's here, so under it the late thread's
    # answer to the second dump can be lost; it then counts as missing there too.
    (first, first_missing), (second, second_missing) = dumps
    late = named(second, "late-answer")
    return (not named(first, "late-answer") and not named(first + second, "never-answers")
            and (len(late) == 1 or (kind == "thread" and not late))
            and first_missing - 2 in extra and second_missing - (1 if late else 2) in extra)


def nested_quit(dumps, extra, kind):
    # The gated thread's answer to the first dump came after its end; the dump it started contained the fault and
    # ended, its block first; a further dump lists it unwound.
    (first, first_missing), (second, second_missing), (third, third_missing) = dumps
    return (not named(first, "gated") and first_missing - 1 in extra
            and len(named(second, "gated")) == 1 and second[0] in named(second, "gated") and FAULT.search(second[0])
            and second_missing in extra and len(named(third, "gated")) == 1 and not FAULT.search(named(third, "gated")[0])
            and third_missing in extra)


CASES = {"full-guards": {"dumps": 1, "check": full_guards},
         "late-answer": {"dumps": 2, "check": late_answer},
         "nested-quit": {"dumps": 3, "check": nested_quit}}

if __name__ == "__main__":
    raise SystemExit(main())
