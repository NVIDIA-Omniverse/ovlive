// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>

namespace omni::usd::live::sceneWork {
// The scene elements (sections, specs, fields and tree nodes) that scene code on the
// calling thread has visited or copied. Scene containers count every element a
// traversal, copy or comparison touches, so a test that reads the meter
// around one host turn measures that turn's work independently of wall time.
// Lookups through a balanced tree or a hash index are not counted.
inline thread_local uint64_t counted = 0;
inline void count(size_t elements) noexcept { counted += elements; }
inline uint64_t total() noexcept { return counted; }
// The key comparisons that ordered scene indexes on the calling thread have
// made while descending: the logarithmic factor of every lookup and edit, which
// the element count above leaves out.
inline thread_local uint64_t compared = 0;
inline void compare() noexcept { ++compared; }
inline uint64_t comparisons() noexcept { return compared; }
// The atomic read-modify-write updates of the owner pins and reentry-depth
// guards that host storage and its projection have taken on the calling
// thread: each counts its increment and its decrement when it is taken, and a
// compare-exchange that retries under contention counts once. Reference
// counts of values a read copies out are not counted.
inline thread_local uint64_t synchronized = 0;
inline void synchronize(size_t updates) noexcept { synchronized += updates; }
inline uint64_t atomics() noexcept { return synchronized; }
} // namespace omni::usd::live::sceneWork
