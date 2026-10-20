// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// The lock half of exit_flush.cpp, Windows only: a critical section that a thread of the process holds until the
// process exits and that this DLL's DLL_PROCESS_DETACH enters, as the detach work of the C++ runtime, USD and other
// dependencies enters locks that a thread ExitProcess terminated can hold.
#include <windows.h>

namespace {
CRITICAL_SECTION lock;
}

// Enters the lock, signals `held` and keeps the lock until ExitProcess ends the calling thread.
extern "C" __declspec(dllexport) void liveExitFlushHoldLock(HANDLE held) {
    EnterCriticalSection(&lock);
    SetEvent(held);
    for (;;) WaitForSingleObject(GetCurrentProcess(), INFINITE); // Signaled only once the process has ended.
}

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) InitializeCriticalSection(&lock);
    if (reason == DLL_PROCESS_DETACH) {
        EnterCriticalSection(&lock);
        LeaveCriticalSection(&lock);
    }
    return TRUE;
}
