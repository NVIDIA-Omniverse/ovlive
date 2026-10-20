// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "AbstractValue.h"
#include <cstdint>
#include <type_traits>

using namespace PXR_NS;
namespace omni::usd::live {
namespace {
// Locally stored, trivially copyable placeholder whose bytes fill the whole
// value storage with zeros. A stock assignment that throws after recording a
// remote type leaves those bytes, a null counted pointer, which destroys safely.
struct Unassigned {
    std::uint64_t zero = 0;
    bool operator==(const Unassigned&) const { return true; }
};
static_assert(sizeof(Unassigned) == sizeof(void*) && std::is_trivially_copyable_v<Unassigned>);
}
bool extractAbstractValue(const SdfAbstractDataConstValue& value, const VtValue& initial, VtValue& output) {
    output = VtValue(Unassigned{});
    if (!value.GetValue(&output)) return false;
    if (output.IsHolding<Unassigned>()) output = initial;
    return true;
}
} // namespace omni::usd::live
