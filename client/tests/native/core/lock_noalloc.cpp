// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "ReaderWriterLock.h"
#include <cstdlib>
#include <iostream>
#include <new>

// A separate executable keeps this allocation fault out of USD and other tests.
static bool rejectAllocation = false;
void* operator new(std::size_t bytes)
{
    if (rejectAllocation) throw std::bad_alloc();
    if (void* result = std::malloc(bytes ? bytes : 1)) return result;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

int main()
{
    threading::ReaderWriterLock<int> a(7), b(7);
    rejectAllocation = true;
    {
        auto read = a.AcquireReadLock();
        auto nestedRead = a.AcquireReadLock();
        auto write = b.AcquireWriteLock();
        auto nestedWrite = b.AcquireWriteLock();
        auto writeRead = b.AcquireReadLock();
        write.unlock(); read.unlock();
        nestedWrite.unlock(); nestedRead.unlock();
    }
    {
        auto write = a.AcquireWriteLock();
        auto writeB = b.AcquireWriteLock();
        auto revisit = a.AcquireReadLock();
        write.unlock(); writeB.unlock();
    }
    rejectAllocation = false;
    // The runner requires this final line; it counts the two scoped blocks above.
    std::cout << "live-cases passed=2" << std::endl;
}
