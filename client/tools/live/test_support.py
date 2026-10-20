# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only scheduling, seed recording, per-case results, and bounded child supervision.

This module is not linked or installed with the live runtime. Virtual scheduling
controls only injected test tasks; it does not claim to reproduce OS threads.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import sys
import threading
import time
import unittest
from typing import Callable, Sequence

from tools.live import process_guard


def write_json(path: Path, value: object) -> None:
    """Publish a complete artifact; the caller owns the destination directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def writer_seed(master: int, writer: int) -> int:
    # Independent of Python's process-randomized hash and Random implementation.
    digest = hashlib.sha256(f"live-seed-v1:{master}:{writer}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def record_seeds(directory: Path, master: int, writers: int) -> dict:
    if master < 0 or writers < 1:
        raise ValueError("nonnegative master seed and positive writer count required")
    seeds = {"schema_version": 1, "master_seed": master,
             "writer_seeds": [writer_seed(master, i) for i in range(writers)]}
    write_json(directory / "seeds.json", seeds)
    # Must precede process creation so a crash still leaves reproduction inputs.
    print(json.dumps(seeds, sort_keys=True), flush=True)
    return seeds


class RecordingResult(unittest.TextTestResult):
    """A text result that also records each case's outcome for JUnit and evidence files."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cases: list[dict] = []
        self._failed_subtests: set[str] = set()

    def _record(self, test, outcome, message=""):
        self.cases.append({"id": test.id(), "outcome": outcome, "message": message})

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "passed")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "failed", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "error", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "skipped", reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._record(test, "expected_failure", self._exc_info_to_string(err, test))

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._record(test, "unexpected_success")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        # A failed subtest suppresses addSuccess; record the enclosing case once.
        if err is not None and test.id() not in self._failed_subtests:
            self._failed_subtests.add(test.id())
            outcome = "failed" if issubclass(err[0], test.failureException) else "error"
            self._record(test, outcome, self._exc_info_to_string(err, test))


def test_cases(suite):
    """The test cases of a unittest suite, in run order."""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from test_cases(item)
        else:
            yield item


def parse_shard(text: str) -> tuple[int, int]:
    """`I/N` with 0 <= I < N as (I, N)."""
    match = re.fullmatch(r"(0|[1-9][0-9]*)/([1-9][0-9]*)", text)
    if not match or int(match.group(1)) >= int(match.group(2)):
        raise ValueError(f"shard must be I/N with 0 <= I < N, not {text!r}")
    return int(match.group(1)), int(match.group(2))


def in_shard(test_id: str, shard: tuple[int, int]) -> bool:
    """A case's shard depends only on its ID, never on discovery order or Python's hash seed."""
    return int.from_bytes(hashlib.sha256(test_id.encode()).digest()[:8], "big") % shard[1] == shard[0]


def case_selection(environment=None) -> tuple[list[str], tuple[int, int] | None]:
    """The scenarios and shard the runner selected (LIVE_TEST_SCENARIOS, a JSON list, and LIVE_TEST_SHARD)."""
    environment = os.environ if environment is None else environment
    scenarios = json.loads(environment.get("LIVE_TEST_SCENARIOS") or "[]")
    if not isinstance(scenarios, list) or not all(isinstance(name, str) and name for name in scenarios):
        raise ValueError("LIVE_TEST_SCENARIOS must be a JSON list of nonempty names")
    shard = environment.get("LIVE_TEST_SHARD")
    return scenarios, parse_shard(shard) if shard else None


def select_cases(suite, scenarios=(), shard=None):
    """The cases of `suite` that match a scenario and fall in the shard, as a new suite.

    A scenario matches a case ID exactly or as a dotted prefix (module, class or
    method). A scenario that matches no discovered case, or a selection that keeps
    no case, is an error, never an empty pass. A module that failed to import stays
    selected so that its error still fails the run.
    """
    cases = list(test_cases(suite))
    def matches(case, scenario):
        return case.id() == scenario or case.id().startswith(scenario + ".")
    unmatched = [scenario for scenario in scenarios if not any(matches(case, scenario) for case in cases)]
    if unmatched:
        raise ValueError("scenario selects no case: " + ", ".join(unmatched))
    chosen = [case for case in cases
              if type(case).__name__ == "_FailedTest" or (
                  (not scenarios or any(matches(case, scenario) for scenario in scenarios))
                  and (shard is None or in_shard(case.id(), shard)))]
    if not chosen:
        raise ValueError("selection keeps no case")
    return unittest.TestSuite(chosen)


def report_ready(ready: bool, reason: str = "", **details) -> None:
    """Publishes a supervised workload's readiness record, when the runner asked for one.

    The record goes to LIVE_TEST_READY_FILE, which is then removed from this
    process's environment so that its own children never report for it.
    """
    path = os.environ.pop("LIVE_TEST_READY_FILE", None)
    if path:
        write_json(Path(path), dict(details, schema_version=1, ready=ready, reason=reason))


# read_ready's result for a Windows record that replace() may still be moving into place.
PUBLISHING = object()
# Consecutive Windows refusals the supervisor reads as publication, about 1 s of 10 ms polls; a longer refusal
# (an ACL denial, a scanner holding the file) is the start's failure.
PUBLICATION_POLLS = 100


def read_ready(path: Path, publishing: bool = False) -> bool | str | object | None:
    """True for a ready record, the reason text for a failed start, None while no record exists.

    With `publishing`, the workload may still be publishing the record: Windows refuses to open a file
    that replace() is moving into place, so that refusal is PUBLISHING, and a later poll reads the
    complete record.
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except PermissionError as error:
        if publishing and sys.platform == "win32":
            return PUBLISHING
        return f"unreadable readiness record: {type(error).__name__}"
    except (OSError, ValueError) as error:
        return f"unreadable readiness record: {type(error).__name__}"
    if not isinstance(record, dict) or record.get("schema_version") != 1 or type(record.get("ready")) is not bool:
        return "invalid readiness record"
    if record["ready"]:
        return True
    return str(record.get("reason") or "service reported a failed start")


def redact(text: str, patterns) -> tuple[str, int]:
    """Replaces each match of the fixture patterns; a pattern's first group, such as `Bearer `, is kept."""
    total = 0
    for pattern in patterns:
        expression = re.compile(pattern)
        keep = expression.groups > 0
        text, count = expression.subn(lambda match: (match.group(1) if keep else "") + "[redacted]", text)
        total += count
    return text, total


def containment_unsupported() -> str | None:
    """Why this platform cannot run tests, or None where a cleanup backend exists."""
    if sys.platform.startswith("linux") or sys.platform in {"win32", "darwin"}:
        return None
    return (f"{sys.platform} is unsupported: the test runner cannot contain or clean up child process trees there "
            "(open item platform)")


def process_cleanup_scope() -> str:
    """The runner's cleanup boundary; macOS workloads must keep descendants in their process group."""
    return "process-group" if sys.platform == "darwin" else "descendant-tree"


def _outcome(code: int) -> str:
    """The outcome of a workload exit status: a signal on Linux and an NTSTATUS error code on Windows are crashes."""
    if code == 0:
        return "passed"
    if code < 0 or (sys.platform == "win32" and code >= 0xC0000000):
        return "crash"
    return "infrastructure_error" if code == 125 else "failed"


def run_recording(suite, verbosity=2, stream=None):
    """Run a unittest suite with the standard text output; return the result and its recorded cases."""
    result = unittest.TextTestRunner(stream=stream, verbosity=verbosity, resultclass=RecordingResult).run(suite)
    return result, result.cases


class ScheduleMismatch(RuntimeError):
    pass


class DeterministicScheduler:
    """A stable FIFO virtual clock with explicit next-task selection and replay."""
    def __init__(self, replay: list[dict] | None = None):
        self.now_ms = 0
        self.history: list[dict] = []
        self._tasks: list[tuple[int, int, str, Callable[[], None]]] = []
        self._next_ordinal = 0
        self._cancelled: set[int] = set()
        self._replay = replay

    def schedule(self, delay_ms: int, boundary: str, action: Callable[[], None]) -> int:
        if delay_ms < 0 or not boundary:
            raise ValueError("nonnegative delay and named boundary required")
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        heapq.heappush(self._tasks, (self.now_ms + delay_ms, ordinal, boundary, action))
        return ordinal

    def cancel(self, ordinal: int) -> None:
        if ordinal < 0 or ordinal >= self._next_ordinal:
            raise ValueError("unknown task")
        self._cancelled.add(ordinal)

    def step(self) -> bool:
        while self._tasks:
            due, ordinal, boundary, action = heapq.heappop(self._tasks)
            if ordinal in self._cancelled:
                self._cancelled.remove(ordinal)
                continue
            self.now_ms = due
            event = {"time_ms": due, "ordinal": ordinal, "boundary": boundary}
            index = len(self.history)
            if self._replay is not None and (index >= len(self._replay) or self._replay[index] != event):
                raise ScheduleMismatch(f"schedule diverged at event {index}: {event}")
            self.history.append(event)
            action()
            return True
        return False

    def run(self, max_steps: int = 10000) -> None:
        for _ in range(max_steps):
            if not self.step():
                if self._replay is not None and len(self.history) != len(self._replay):
                    raise ScheduleMismatch("replay has unconsumed events")
                return
        # Cancelled tasks do not consume execution budget. Validate replay on
        # every successful exit, including exact-budget exhaustion.
        while self._tasks and self._tasks[0][1] in self._cancelled:
            self._cancelled.remove(heapq.heappop(self._tasks)[1])
        if self._tasks:
            raise RuntimeError("virtual scheduler exceeded its step budget")
        if self._replay is not None and len(self.history) != len(self._replay):
            raise ScheduleMismatch("replay has unconsumed events")


@dataclass(frozen=True)
class ProcessResult:
    outcome: str
    returncode: int | None
    elapsed_ms: int
    output_bytes: int
    output_truncated: bool
    cleanup_complete: bool
    stack_dumps: tuple[str, ...] = ()
    ready: bool | None = None  # None when no readiness was requested.
    start_failure: str = ""


def _handler_installed(pid: int, signum: int) -> bool:
    """Whether /proc/<pid>/status shows a caught handler for signum (process_guard.py reads it the same way)."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("SigCgt:"):
                return bool(int(line.split()[1], 16) >> (signum - 1) & 1)
    except (OSError, ValueError, IndexError):
        pass
    return False


def _dump_files(prefix: Path) -> list[Path]:
    """The <prefix>.<pid> stack dump files, sorted."""
    return sorted(path for path in prefix.parent.glob(prefix.name + ".*") if path.suffix[1:].isdigit())


def _request_stacks(child: subprocess.Popen, prefix: Path) -> None:
    """Asks the Linux guard for its descendants' thread stacks and waits for its <prefix>.signalled record."""
    if not sys.platform.startswith("linux") or child.poll() is not None or not _handler_installed(child.pid, signal.SIGUSR1):
        return  # A guard without its handler yet has started no workload worth dumping.
    signalled = Path(f"{prefix}.signalled")
    os.kill(child.pid, signal.SIGUSR1)
    deadline = time.monotonic() + 5  # Failure-only: the guard waits at most 3 s for dumps.
    while not signalled.is_file():
        if time.monotonic() >= deadline or child.poll() is not None:
            return
        time.sleep(0.01)


def _kill_group(child: subprocess.Popen, status: int | None, stop=None) -> bool:
    """Ends the workload's cleanup scope. A guard reports descendant-tree cleanup on `status`; macOS passes None
    for process-group cleanup. On Windows `stop` is the guard's stop event."""
    if status is None:
        # The macOS workload leads its own process group. Signal the group even
        # when its leader has exited so ordinary children cannot hold the log open.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return False
        return True
    try:
        if child.poll() is None:
            if stop is not None:
                process_guard.set_event(stop)  # The guard terminates its job, even descendants started detached.
            else:
                # A SIGTERM that precedes the guard's handler would end it without its report. The guard installs
                # the handler before it launches the workload; the deadline is failure-only.
                deadline = time.monotonic() + 5
                while (child.poll() is None and not _handler_installed(child.pid, signal.SIGTERM)
                       and time.monotonic() < deadline):
                    time.sleep(0.01)
                child.terminate()  # The guard kills and reaps even descendants in other sessions.
        try:
            child.wait(timeout=6)
            return os.read(status, 64) == b"contained\n"  # Only the exited guard held the write end.
        except subprocess.TimeoutExpired:
            if stop is not None:
                child.kill()  # Closing the guard's job handle kills every member.
            else:
                os.killpg(child.pid, signal.SIGKILL)
            return False
    finally:
        os.close(status)
        if stop is not None:
            process_guard.close_handle(stop)


def run_process(command: Sequence[str], *, cwd: Path, output: Path,
                timeout_ms: int, max_output_bytes: int = 1024 * 1024,
                environment: dict[str, str] | None = None, stack_dump_prefix: Path | None = None,
                expire: threading.Event | None = None, ready: Path | None = None,
                startup_timeout_ms: int | None = None, scratch: bool = False) -> ProcessResult:
    """Drain output independently of bounded storage; kill descendants on exit.

    On Linux the workload receives LIVE_STACK_DUMP_PREFIX (default: output with
    the suffix .stacks), made absolute against this process's working
    directory; a watchdog expiry first asks every descendant with a dump
    handler for its thread stacks. The run owns its prefix: dumps and a
    .signalled record left by an earlier run are removed before launch, and
    every non-empty <prefix>.<pid> dump is listed in stack_dumps whatever the
    outcome, including one an in-process watchdog wrote before aborting.
    `expire` is a test seam that takes the timeout path when set.

    With `ready`, the workload is a service fixture: it receives
    LIVE_TEST_READY_FILE and publishes one record there (report_ready) once its
    service answers, or with a reason when its start fails. The runner waits for
    that record; an exit before a ready record, a failed-start record or no
    record within `startup_timeout_ms` (a failure-only watchdog, default the
    whole timeout) is the outcome `service_start_failed`, never a pass.

    With `scratch`, the workload gets a fresh `live-scratch-*` directory as
    LIVE_TEST_SCRATCH_DIR, TMPDIR, TMP and TEMP. The directory is removed with the
    process tree: by this function when the run ends or is interrupted, and by
    the guard when this process dies first on Linux and Windows. macOS uses a
    workload process group, which cannot contain descendants that detach or
    guarantee cleanup after the runner is forcibly killed. Cleanup is incomplete
    while the scratch directory remains.

    On Linux and Windows the workload runs under an independent guard: a subreaper on Linux, which
    reaps even descendants in other sessions, and a Job Object on Windows, which
    every descendant joins before it runs. Watchdog stack dumps are Linux-only.
    macOS runs the workload directly in its own process group. Other platforms
    launch nothing and return `infrastructure_error`.

    Commands are argv arrays, never shell text. Callers supply synthetic test
    inputs and a credential-free environment. Output may contain test values;
    real-service adapters must apply their stricter artifact redaction policy.
    """
    startup_ms = min(timeout_ms if startup_timeout_ms is None else startup_timeout_ms, timeout_ms)
    if timeout_ms <= 0 or max_output_bytes < 0 or not command or startup_ms <= 0:
        raise ValueError("positive timeouts, nonnegative output budget, and command required")
    unsupported = containment_unsupported()
    if unsupported:
        output.write_text(unsupported + "\n", encoding="utf-8")
        return ProcessResult("infrastructure_error", None, 0, 0, False, True)
    started = time.monotonic()
    counts = {"total": 0, "stored": 0}
    prefix = (Path(stack_dump_prefix) if stack_dump_prefix is not None else output.with_suffix(".stacks")).absolute()
    for path in (*_dump_files(prefix), Path(f"{prefix}.signalled")):
        path.unlink(missing_ok=True)
    environment = dict(os.environ if environment is None else environment)
    if ready is not None:
        ready = Path(ready).absolute()
        ready.unlink(missing_ok=True)
        environment["LIVE_TEST_READY_FILE"] = str(ready)
    # Resolved, so the workload's temporary paths have one spelling: Windows may name the system temporary
    # directory in its short 8.3 form, which resolve() expands.
    scratch_directory = Path(tempfile.mkdtemp(prefix="live-scratch-")).resolve() if scratch else None
    if scratch_directory is not None:
        environment.update(LIVE_TEST_SCRATCH_DIR=str(scratch_directory), TMPDIR=str(scratch_directory),
                           TMP=str(scratch_directory), TEMP=str(scratch_directory))

    def remove_scratch() -> bool:
        if scratch_directory is None:
            return True
        shutil.rmtree(scratch_directory, ignore_errors=True)
        return not scratch_directory.exists()

    stop = None
    try:
        # SDK LD_LIBRARY_PATH/PYTHONHOME settings belong to the workload.
        # Applying them to our supervisor can replace its libpython and
        # remove the pidfd APIs needed for reliable descendant containment.
        # Use an anonymous private descriptor, never argv or a named artifact,
        # to hand the exact workload environment to the independent guard.
        workload_environment = dict(environment, LIVE_STACK_DUMP_PREFIX=str(prefix))
        payload = json.dumps(workload_environment).encode()
        if len(payload) > 1024 * 1024:
            raise ValueError("workload environment exceeds 1 MiB")
        guard_environment = {key: value for key, value in os.environ.items()
                             if key in {"PATH", "LD_LIBRARY_PATH", "SYSTEMROOT", "WINDIR",
                                        "COMSPEC", "PATHEXT", "TEMP", "TMP"}}
        guard_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        status, status_write = (None, None) if sys.platform == "darwin" else os.pipe()
        try:
            with tempfile.TemporaryFile() as configuration:
                configuration.write(payload)
                configuration.seek(0)
                # A Windows venv's python.exe is a launcher that runs the base interpreter as its child, which would
                # make the launcher, not this process, the guard's parent; the guard needs only the standard library.
                interpreter = sys._base_executable if sys.platform == "win32" else sys.executable
                guard = [interpreter, "-I", "-B", str(Path(__file__).with_name("process_guard.py")), str(os.getpid())]
                if sys.platform == "darwin":
                    child = subprocess.Popen(command, cwd=cwd, env=workload_environment,
                                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                elif sys.platform == "win32":
                    # The guard inherits exactly its configuration, status and stop handles.
                    import msvcrt
                    handles = [msvcrt.get_osfhandle(configuration.fileno()), msvcrt.get_osfhandle(status_write)]
                    for handle in handles:
                        os.set_handle_inheritable(handle, True)
                    stop = process_guard.create_stop_event()
                    handles.append(stop)
                    child = subprocess.Popen(
                        [*guard, *map(str, handles), str(scratch_directory or ""), *command], cwd=cwd,
                        env=guard_environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, close_fds=True,
                        startupinfo=subprocess.STARTUPINFO(lpAttributeList={"handle_list": handles}),
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                else:
                    child = subprocess.Popen(
                        [*guard, str(configuration.fileno()), str(status_write), str(scratch_directory or ""), *command],
                        cwd=cwd, env=guard_environment, pass_fds=(configuration.fileno(), status_write),
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        start_new_session=True)
        except BaseException:
            if status is not None:
                os.close(status)
            if stop is not None:
                process_guard.close_handle(stop)
            raise
        finally:
            if status_write is not None:
                os.close(status_write)
    except OSError as error:
        output.write_text(f"process launch failed: {type(error).__name__}\n", encoding="utf-8")
        return ProcessResult("infrastructure_error", None, 0, 0, False, remove_scratch())
    except BaseException:
        remove_scratch()
        raise

    reader_errors: list[Exception] = []

    def drain() -> None:
        try:
            with output.open("wb") as log:
                assert child.stdout is not None
                while True:
                    chunk = child.stdout.read1(65536)  # Returns what has arrived, not a full block.
                    if not chunk:
                        break
                    counts["total"] += len(chunk)
                    remaining = max_output_bytes - counts["stored"]
                    if remaining > 0:
                        saved = chunk[:remaining]
                        log.write(saved)
                        log.flush()  # The log is readable while the process runs.
                        counts["stored"] += len(saved)
        except Exception as error:
            reader_errors.append(error)

    reader = threading.Thread(target=drain, name="test-output-drain", daemon=True)
    reader.start()
    outcome = "infrastructure_error"
    deadline = started + timeout_ms / 1000
    startup_deadline = started + startup_ms / 1000
    readiness = None  # True once ready, the failure reason once a start failed.
    refusals = 0  # Consecutive Windows refusals to open a record being published.
    start_failure = ""
    try:
        try:
            code = None
            while code is None:
                if expire is not None and expire.is_set():
                    raise subprocess.TimeoutExpired(list(command), timeout_ms / 1000)
                if ready is not None and readiness is None:
                    readiness = read_ready(ready, publishing=refusals < PUBLICATION_POLLS)
                    if readiness is PUBLISHING:
                        refusals, readiness = refusals + 1, None
                    elif readiness is None:
                        refusals = 0
                    if readiness is None and time.monotonic() >= startup_deadline:
                        start_failure = f"no readiness record within {startup_ms} ms"
                        raise subprocess.TimeoutExpired(list(command), timeout_ms / 1000)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(list(command), timeout_ms / 1000)
                # Poll the observable readiness record and the expiry seam; otherwise wait for the exit.
                polling = expire is not None or (ready is not None and readiness is None)
                try:
                    code = child.wait(timeout=min(remaining, 0.01) if polling else remaining)
                except subprocess.TimeoutExpired:
                    pass
            outcome = _outcome(code)
        except subprocess.TimeoutExpired:
            outcome = "timeout"
            _request_stacks(child, prefix)
    finally:
        # Also terminate a grandchild left behind by an otherwise successful test.
        contained = _kill_group(child, status, stop)
        child.wait(timeout=5)
        reader.join(timeout=5)
        if not reader.is_alive() and child.stdout is not None:
            child.stdout.close()
        scratch_removed = remove_scratch()
    if ready is not None:
        if readiness is None:
            readiness = read_ready(ready)  # A record published just before the exit.
        if readiness is not True:
            start_failure = start_failure or (readiness if isinstance(readiness, str) else
                                              f"exited with {child.returncode} before reporting readiness")
            outcome = "service_start_failed"
    cleanup_complete = contained and not reader.is_alive() and scratch_removed
    if reader_errors or not cleanup_complete:
        outcome = "infrastructure_error"
    # Processes with a handler that never dumped leave empty files.
    stack_dumps = []
    for path in _dump_files(prefix):
        try:
            if path.stat().st_size == 0:
                path.unlink()
            else:
                stack_dumps.append(str(path))
        except OSError:
            pass
    return ProcessResult(outcome, child.returncode, int((time.monotonic() - started) * 1000),
                         counts["total"], counts["total"] > counts["stored"], cleanup_complete, tuple(stack_dumps),
                         None if ready is None else readiness is True, start_failure)
