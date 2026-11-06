# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture a private minidump of a supervised Windows test child that failed to exit; never a success verdict."""
import ctypes
from ctypes import wintypes
import json
import os
import shutil
from pathlib import Path
import sys


def capture(pid, path):
    if os.name != 'nt' or pid <= 0:
        raise ValueError('A Windows child PID is required')
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    debug = ctypes.WinDLL('dbghelp', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    debug.MiniDumpWriteDump.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.HANDLE, wintypes.DWORD,
                                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    debug.MiniDumpWriteDump.restype = wintypes.BOOL
    # Query information, read memory and duplicate handles of this test's own supervised child; no debug privilege.
    process = kernel.OpenProcess(0x0400 | 0x0010 | 0x0040, False, pid)
    if not process:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        with Path(path).open('xb') as stream:
            # Normal thread stacks/registers plus thread information, without a full-memory dump.
            if not debug.MiniDumpWriteDump(process, pid, msvcrt.get_osfhandle(stream.fileno()), 0x1000,
                                          None, None, None):
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(process)


if __name__ == '__main__':
    capture(int(sys.argv[1]), sys.argv[2])
    cdb = shutil.which('cdb')
    sdk_cdb = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Windows Kits/10/Debuggers/x64/cdb.exe'
    print(json.dumps(dict(pid=int(sys.argv[1]), dump=str(Path(sys.argv[2]).name),
                          cdb=cdb or (str(sdk_cdb) if sdk_cdb.is_file() else None))))
