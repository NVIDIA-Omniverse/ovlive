// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// The process half of tests/harness/test_exit_flush.py. It writes one record of more than three of the C runtime's
// 4 KiB stdout buffers to a pipe and returns from main, the way a native test driver or a sample ends. On Windows
// another thread first takes the lock of exit_lock.cpp, so ExitProcess terminates it while it holds the lock; that
// DLL's process detach then enters the orphaned lock and Windows ends the process at once with exit code 0, before
// the C runtime's own process detach flushes stdout. Modes:
//   crt     the record waits in the C runtime's buffer for that flush;
//   driver  the record is written as a native test driver writes, under the stdout policy of
//           tests/native/support/StackDump.cpp, which every driver links;
//   sample  the record is written with writeRecord, as the diagnostics sample writes each record.
#include "RecordWriter.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#if defined(_WIN32)
#    include <windows.h>
extern "C" __declspec(dllimport) void liveExitFlushHoldLock(HANDLE held);
#endif

int main(int argc, char** argv) {
    const std::string mode = argc == 2 ? argv[1] : "";
    if (mode != "crt" && mode != "driver" && mode != "sample") return 2;
    // StackDump.cpp's policy applies before main; the other modes restore the C runtime's default for a pipe.
    if (mode != "driver" && std::setvbuf(stdout, nullptr, _IOFBF, 4096) != 0) return 2;
#if defined(_WIN32)
    HANDLE held = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!held) return 2;
    std::thread([held] { liveExitFlushHoldLock(held); }).detach();
    if (WaitForSingleObject(held, INFINITE) != WAIT_OBJECT_0) return 2;
#endif
    const std::string record = std::string(13000, 'x') + " end\n";
    if (mode == "driver") std::cout << record;
    else if (mode == "sample") return omni::usd::live::sample::writeRecord(stdout, record) ? 0 : 1;
    else std::fwrite(record.data(), 1, record.size(), stdout);
    return 0;
}
