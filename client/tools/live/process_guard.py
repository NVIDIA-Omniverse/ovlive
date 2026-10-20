# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-process containment, including children which create new sessions or detach.

Each workload gets its own guard: a subreaper on Linux, which owns every
orphaned descendant until it is killed and reaped, and a Job Object on Windows,
which the workload joins while suspended, so every descendant is a member. The
workload's scratch directory is removed after them, also when the runner dies
first. This is test infrastructure, never shipped.
"""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


class StopRequested(BaseException):
    pass


def stop_requested(signum, frame):
    raise StopRequested()


def process_table():
    records = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm can contain spaces/parentheses; fields after its last ')' are
            # state, parent, group, ..., starttime (Linux proc_pid_stat).
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            records[int(entry.name)] = (int(fields[1]), fields[0], fields[19])
        except (OSError, ValueError, IndexError):
            continue
    return records


def descendants(root, table):
    owned = {root}
    while True:
        added = {pid for pid, (parent, _, _) in table.items() if parent in owned} - owned
        if not added:
            return owned - {root}
        owned.update(added)


def handler_installed(pid, signum):
    """Whether /proc/<pid>/status shows a caught handler for signum."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("SigCgt:"):
                return bool(int(line.split()[1], 16) >> (signum - 1) & 1)
    except (OSError, ValueError, IndexError):
        pass
    return False


def dump_prefix(pid):
    """The process's LIVE_STACK_DUMP_PREFIX, or None when absent or unreadable."""
    try:
        for entry in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
            if entry.startswith(b"LIVE_STACK_DUMP_PREFIX="):
                return entry.split(b"=", 1)[1].decode() or None
    except (OSError, UnicodeError):
        pass
    return None


def request_stacks(prefix):
    """Runner SIGUSR1: ask each descendant with a dump handler for its thread
    stacks, wait at most 3 s, and record them in <prefix>.signalled.

    A target has LIVE_STACK_DUMP_PREFIX in its environment and a caught
    SIGQUIT in SigCgt. SigCgt proves some SIGQUIT handler, not the dump
    handler: a workload that inherits the prefix and installs its own SIGQUIT
    handler receives SIGQUIT on a watchdog expiry too."""
    table = process_table()
    targets = []
    for pid in sorted(descendants(os.getpid(), table)):
        target = dump_prefix(pid)
        if not target or not handler_installed(pid, signal.SIGQUIT):
            continue  # Nested guards and processes without a handler keep their default action.
        try:
            descriptor = os.pidfd_open(pid)
            try:
                current = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                if current[19] != table[pid][2]:
                    continue
                signal.pidfd_send_signal(descriptor, signal.SIGQUIT)
            finally:
                os.close(descriptor)
        except OSError:
            continue
        targets.append(dict(pid=pid, file=f"{target}.{pid}"))

    def complete(entry):
        try:
            text = Path(entry["file"]).read_bytes()
        except OSError:
            return False
        # A native dump ends with its end line; a Python dump is written in one handler call.
        return bool(text) and (not text.startswith(b"live-stack-dump") or b"live-stack-dump end" in text)
    deadline = time.monotonic() + 3  # Failure-only: an unfinished dump is recorded as incomplete.
    while time.monotonic() < deadline and not all(complete(entry) for entry in targets):
        time.sleep(0.01)
    for entry in targets:
        entry["complete"] = complete(entry)
    staged = Path(f"{prefix}.signalled.tmp")
    staged.write_text(json.dumps(targets) + "\n")
    staged.replace(f"{prefix}.signalled")


def _safe_request(prefix):
    # Containment must never fail because a stack dump could not be requested.
    if not prefix:
        return
    try:
        request_stacks(prefix)
    except Exception:
        pass


def cleanup():
    # This guard, unlike its workload, stays alive until all descendants are
    # gone. Reparenting therefore cannot hide a nested session or double fork.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        table = process_table()
        owned = descendants(os.getpid(), table)
        for pid in owned:
            try:
                descriptor = os.pidfd_open(pid)
                try:
                    # A PID can be reused between enumeration and opening. A
                    # pidfd pins the process; revalidate its birth time before
                    # signaling so unrelated processes can never be targeted.
                    current = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                    if current[19] == table[pid][2]:
                        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                finally:
                    os.close(descriptor)
            except (ProcessLookupError, FileNotFoundError):
                pass
        # Only this dedicated guard's children are reaped, not runner siblings.
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
            except ChildProcessError:
                return True
        time.sleep(0.005)
    return False


def remove_scratch(scratch):
    """Removes the `live-scratch-*` directory the runner handed this guard once no descendant can write to it.

    The path comes from the guard's own arguments, never from the workload
    environment, which a nested run inherits from the suite that owns it."""
    if not scratch:
        return True
    if not Path(scratch).name.startswith("live-scratch-"):
        return False
    shutil.rmtree(scratch, ignore_errors=True)
    return not Path(scratch).exists()


# Windows: the guard joins its workload to a Job Object. The structures and functions below are used only there.
SYNCHRONIZE = 0x00100000
WAIT_OBJECT_0 = 0
INFINITE = 0xFFFFFFFF
CREATE_SUSPENDED = 0x00000004
SEM_FAILCRITICALERRORS = 0x0001
SEM_NOGPFAULTERRORBOX = 0x0002
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


class _BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", ctypes.c_uint64 * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]


class _Security(ctypes.Structure):
    _fields_ = [("nLength", wintypes.DWORD), ("lpSecurityDescriptor", ctypes.c_void_p),
                ("bInheritHandle", wintypes.BOOL)]


# kernel32 function, result type and argument types.
_SIGNATURES = (
    ("CreateJobObjectW", wintypes.HANDLE, [ctypes.c_void_p, wintypes.LPCWSTR]),
    ("SetInformationJobObject", wintypes.BOOL, [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]),
    ("QueryInformationJobObject", wintypes.BOOL,
     [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]),
    ("AssignProcessToJobObject", wintypes.BOOL, [wintypes.HANDLE, wintypes.HANDLE]),
    ("TerminateJobObject", wintypes.BOOL, [wintypes.HANDLE, wintypes.UINT]),
    ("OpenProcess", wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]),
    ("WaitForSingleObject", wintypes.DWORD, [wintypes.HANDLE, wintypes.DWORD]),
    ("WaitForMultipleObjects", wintypes.DWORD,
     [wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE), wintypes.BOOL, wintypes.DWORD]),
    ("CloseHandle", wintypes.BOOL, [wintypes.HANDLE]),
    ("CreateEventW", wintypes.HANDLE, [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]),
    ("SetEvent", wintypes.BOOL, [wintypes.HANDLE]),
    ("SetErrorMode", wintypes.UINT, [wintypes.UINT]),
)
if sys.platform == "win32":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")
    for _name, _result, _arguments in _SIGNATURES:
        getattr(_kernel32, _name).restype, getattr(_kernel32, _name).argtypes = _result, _arguments
    _ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]


def _check(succeeded):
    if not succeeded:
        raise ctypes.WinError(ctypes.get_last_error())
    return succeeded


def create_stop_event():
    """An inheritable, unsignaled manual-reset event the runner sets to stop its guard."""
    security = _Security(ctypes.sizeof(_Security), None, True)
    return _check(_kernel32.CreateEventW(ctypes.byref(security), True, False, None))


def set_event(handle):
    _check(_kernel32.SetEvent(handle))


def close_handle(handle):
    _kernel32.CloseHandle(handle)


def _job():
    """A job whose last closed handle kills every member, and whose members die on an unhandled exception rather than
    waiting on an error report dialog."""
    job = _check(_kernel32.CreateJobObjectW(None, None))
    limits = _ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                                               | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION)
    _check(_kernel32.SetInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits),
                                             ctypes.sizeof(limits)))
    return job


def _empty(job):
    """Terminates every job member and waits at most 5 s for none to remain."""
    _kernel32.TerminateJobObject(job, 1)
    accounting = _Accounting()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        _check(_kernel32.QueryInformationJobObject(job, JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
                                                   ctypes.byref(accounting), ctypes.sizeof(accounting), None))
        if accounting.ActiveProcesses == 0:
            return True
        time.sleep(0.005)
    return False


def _windows_environment(handle):
    """The workload environment the runner wrote to the configuration handle, or None when it is invalid."""
    import msvcrt
    try:
        with os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY), "rb") as source:
            payload = source.read(1024 * 1024 + 1)
        environment = json.loads(payload) if len(payload) <= 1024 * 1024 else None
    except (OSError, ValueError, UnicodeError):
        return None
    # A Windows environment block may hold per-drive entries such as `=C:`, whose name starts with `=`.
    if (not isinstance(environment, dict)
            or any(not isinstance(key, str) or not isinstance(value, str) or not key or "=" in key[1:]
                   or "\0" in key or "\0" in value for key, value in environment.items())):
        return None
    return environment


def _run_in_job(job, command, environment, stop, runner):
    """Runs the workload as a job member; its exit status, or 1 when the stop event or the runner's exit ends it."""
    # The workload starts suspended and joins the job before it runs, so no descendant can start outside it. It
    # inherits an error mode in which a DLL the loader cannot find fails the start instead of raising a dialog.
    _kernel32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX)
    child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, creationflags=CREATE_SUSPENDED)
    try:
        _check(_kernel32.AssignProcessToJobObject(job, int(child._handle)))
        if _ntdll.NtResumeProcess(int(child._handle)) != 0:
            raise OSError("NtResumeProcess failed")
        handles = (wintypes.HANDLE * 3)(int(child._handle), stop, runner)
        if _kernel32.WaitForMultipleObjects(3, handles, False, INFINITE) == WAIT_OBJECT_0:
            return child.wait()
        return 1
    except BaseException:
        _kernel32.TerminateJobObject(job, 1)
        raise


def windows_main():
    # argv: runner PID, configuration handle, status handle, stop event handle, scratch directory or "", workload
    # command. The runner passes exactly these three handles; nothing else is inherited.
    import msvcrt
    if len(sys.argv) < 7:
        return 125
    environment = _windows_environment(int(sys.argv[2]))
    if environment is None:
        print("test process guard: invalid workload environment", file=sys.stderr)
        return 125
    expected_parent = int(sys.argv[1])
    status = msvcrt.open_osfhandle(int(sys.argv[3]), os.O_WRONLY)
    stop, scratch = int(sys.argv[4]), sys.argv[5]
    # The runner's process handle stands in for Linux's parent-death signal: its exit stops this guard.
    runner = _kernel32.OpenProcess(SYNCHRONIZE, False, expected_parent)
    if not runner or os.getppid() != expected_parent:
        return 125
    result = 125
    job = _job()
    try:
        result = _run_in_job(job, sys.argv[6:], environment, stop, runner)
    except OSError:
        result = 125
    finally:
        contained = _empty(job)
        _kernel32.CloseHandle(job)
        _kernel32.CloseHandle(runner)
        # The runner reads containment from the status handle, never from the exit status, which stays the workload's
        # own.
        if contained and remove_scratch(scratch):
            os.write(status, b"contained\n")
        else:
            result = 125
        os.close(status)
    return result


def _wait_exit_windows(pid, timeout_s):
    handle = _kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return True
    try:
        return _kernel32.WaitForSingleObject(handle, int(timeout_s * 1000)) == WAIT_OBJECT_0
    finally:
        _kernel32.CloseHandle(handle)


def _wait_exit_linux(pid, timeout_s):
    import select
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return True
    try:
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        return bool(poller.poll(int(timeout_s * 1000)))
    finally:
        os.close(descriptor)


def wait_exit(pid, timeout_s):
    """Whether process `pid` has exited, waiting at most `timeout_s`. A process that does not exist has exited, and so
    has an exited one its parent has not yet reaped."""
    return (_wait_exit_windows if sys.platform == "win32" else _wait_exit_linux)(pid, timeout_s)


def alive(pid):
    """Whether process `pid` exists, counting an unreaped exited one on Linux; on Windows this never signals it, as
    os.kill(pid, 0) would terminate it."""
    if sys.platform == "win32":
        return not wait_exit(pid, 0)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main():
    # argv: runner PID, configuration descriptor, status descriptor, scratch directory or "", workload command.
    if sys.platform == "win32":
        return windows_main()
    if not sys.platform.startswith("linux") or len(sys.argv) < 6:
        return 125
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        print("test process guard: supervisor Python lacks Linux pidfd support", file=sys.stderr)
        return 125
    try:
        with os.fdopen(int(sys.argv[2]), "rb") as source:
            payload = source.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            return 125
        environment = json.loads(payload)
        if (not isinstance(environment, dict) or
                any(not isinstance(key, str) or not isinstance(value, str) or
                    not key or "=" in key or "\0" in key or "\0" in value
                    for key, value in environment.items())):
            return 125
    except (OSError, ValueError, UnicodeError):
        print("test process guard: invalid workload environment", file=sys.stderr)
        return 125
    try:
        probe = os.pidfd_open(os.getpid())
        os.close(probe)
    except OSError:
        return 125
    expected_parent = int(sys.argv[1])
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_CHILD_SUBREAPER=36; PR_SET_PDEATHSIG=1. The latter requests cleanup
    # if the owning runner dies. Check parent after installation to close race.
    if libc.prctl(36, 1, 0, 0, 0) != 0 or libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        return 125
    if os.getppid() != expected_parent:
        return 125
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    status, scratch = int(sys.argv[3]), sys.argv[4]
    prefix = environment.get("LIVE_STACK_DUMP_PREFIX")
    signal.signal(signal.SIGUSR1, lambda signum, frame: _safe_request(prefix))
    result = 125
    try:
        child = subprocess.Popen(sys.argv[5:], env=environment, stdin=subprocess.DEVNULL)
        result = child.wait()
    except StopRequested:
        result = -signal.SIGTERM
    except OSError:
        result = 125
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        # The runner reads containment from the status descriptor, never from the
        # exit status, which stays the workload's own.
        if cleanup() and remove_scratch(scratch):
            os.write(status, b"contained\n")
        else:
            result = 125
    if result < 0:
        # Preserve a crashed workload's signal classification for the parent.
        signum = -result
        if signum not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
