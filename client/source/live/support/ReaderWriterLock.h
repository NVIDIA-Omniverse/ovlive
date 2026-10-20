// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstdint>
#include <shared_mutex>
#include <system_error>
#include <thread>
#include <utility>

namespace threading {
// Reentrancy belongs to this lock's address, never to the caller's spec ID.
// Read/read, write/write and write/read nesting are supported across distinct
// instances of the same T. Read-to-write upgrades fail instead of deadlocking.
// All guards must retire on their acquiring thread, before the owner is destroyed.
// The outermost acquisition holds the underlying mutex until every nested guard
// retires, including when callers explicitly unlock guards out of order.
template<typename T> class ReaderWriterLock;
template<typename T> struct LockOwnership {
    const ReaderWriterLock<T>* owner = nullptr;
    LockOwnership* previous = nullptr;
    LockOwnership* next = nullptr;
    std::thread::id thread;
    bool exclusive = false;
};

template<typename T> class SharedLock {
public:
    ~SharedLock() noexcept { unlock(); }
    SharedLock(const SharedLock&) = delete;
    SharedLock(SharedLock&&) = delete;
    SharedLock& operator=(const SharedLock&) = delete;
    void unlock() noexcept;
private:
    friend class ReaderWriterLock<T>;
    explicit SharedLock(const ReaderWriterLock<T>* owner);
    LockOwnership<T> _held;
};

template<typename T> class ScopedLock {
public:
    ~ScopedLock() noexcept { unlock(); }
    ScopedLock(const ScopedLock&) = delete;
    ScopedLock(ScopedLock&&) = delete;
    ScopedLock& operator=(const ScopedLock&) = delete;
    void unlock() noexcept;
private:
    friend class ReaderWriterLock<T>;
    explicit ScopedLock(ReaderWriterLock<T>* owner);
    LockOwnership<T> _held;
};

inline std::atomic<uint64_t> nextReaderWriterLockId{1};

template<typename T> class ReaderWriterLock {
public:
    using SharedConditional_t = std::pair<SharedLock<T>, bool>;
    using ScopedConditional_t = std::pair<ScopedLock<T>, bool>;
    ReaderWriterLock() : _id(nextReaderWriterLockId.fetch_add(1, std::memory_order_relaxed)) {}
    explicit ReaderWriterLock(uint64_t id) : _id(id) {}
    ReaderWriterLock(const ReaderWriterLock&) = delete;
    ReaderWriterLock(ReaderWriterLock&&) = delete;
    ReaderWriterLock& operator=(const ReaderWriterLock&) = delete;

    ScopedLock<T> AcquireWriteLock(const char* operation = nullptr);
    SharedLock<T> AcquireReadLock(const char* operation = nullptr) const;
    template<typename V, typename... TArgs> ScopedLock<T> AcquireWriteLock(const char*, const V&, const TArgs&...) {
        return AcquireWriteLock();
    }
    template<typename V, typename... TArgs>
    SharedLock<T> AcquireReadLock(const char*, const V&, const TArgs&...) const {
        return AcquireReadLock();
    }
    uint64_t Id() const { return _id; }

private:
    friend class ScopedLock<T>;
    friend class SharedLock<T>;
    void acquire(LockOwnership<T>& held, bool writer) const;
    void release(LockOwnership<T>& held) const noexcept;
    mutable std::shared_mutex _mutex;
    const uint64_t _id;
    inline static thread_local LockOwnership<T>* ownership = nullptr;
};
} // namespace threading

#include "ReaderWriterLock.inl"
