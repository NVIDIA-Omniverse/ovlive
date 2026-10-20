// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ReaderWriterLock.h"
#include <exception>

namespace threading {
template<typename T> void ReaderWriterLock<T>::acquire(LockOwnership<T>& held, bool writer) const {
    auto existing = ownership;
    while (existing && existing->owner != this) existing = existing->next;
    if (existing) {
        if (writer && !existing->exclusive)
            throw std::system_error(std::make_error_code(std::errc::resource_deadlock_would_occur));
        held.exclusive = existing->exclusive;
    } else {
        if (writer) _mutex.lock();
        else _mutex.lock_shared();
        held.exclusive = writer;
    }
    // Guard storage itself is the ownership record: no allocation on acquisition
    // or release, including through existing noexcept scene swap operations.
    held.owner = this;
    held.thread = std::this_thread::get_id();
    held.next = ownership;
    if (ownership) ownership->previous = &held;
    ownership = &held;
}

template<typename T> void ReaderWriterLock<T>::release(LockOwnership<T>& held) const noexcept {
    if (held.thread != std::this_thread::get_id()) std::terminate();
    if (held.previous) held.previous->next = held.next;
    else ownership = held.next;
    if (held.next) held.next->previous = held.previous;
    held.owner = nullptr;
    for (auto existing = ownership; existing; existing = existing->next)
        if (existing->owner == this) return;
    if (held.exclusive) _mutex.unlock();
    else _mutex.unlock_shared();
}

template<typename T> ScopedLock<T> ReaderWriterLock<T>::AcquireWriteLock(const char*) { return ScopedLock<T>(this); }

template<typename T> SharedLock<T> ReaderWriterLock<T>::AcquireReadLock(const char*) const {
    return SharedLock<T>(this);
}

template<typename T> ScopedLock<T>::ScopedLock(ReaderWriterLock<T>* owner) { owner->acquire(_held, true); }

template<typename T> SharedLock<T>::SharedLock(const ReaderWriterLock<T>* owner) { owner->acquire(_held, false); }

template<typename T> void ScopedLock<T>::unlock() noexcept {
    if (_held.owner) _held.owner->release(_held);
}

template<typename T> void SharedLock<T>::unlock() noexcept {
    if (_held.owner) _held.owner->release(_held);
}
} // namespace threading
