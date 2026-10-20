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
} // namespace omni::usd::live::sceneWork
