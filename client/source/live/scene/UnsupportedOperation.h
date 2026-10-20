// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "UsdValueCodec.h"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace omni::usd::live {
// An unsupported compile that names its cause. `index()` is the position in
// the operation log of the first operation that fails, or noOperation for a
// structural shape no single operation causes. It stays ErrorKind::Unsupported,
// so every DecodeError classification applies unchanged.
class UnsupportedOperation final : public wire::DecodeError {
public:
    enum class Reason : uint32_t {
        ValueType = 1,
        PayloadOffset = 2,
        IdentityTypeChange = 3,
        MultiUpdate = 4,
        MoveCycle = 5,
        TimeSampleDictionaryKey = 6,
        IdentityRealization = 7
    };
    static constexpr size_t noOperation = std::numeric_limits<size_t>::max();
    explicit UnsupportedOperation(Reason reason, size_t index = noOperation)
        : wire::DecodeError(wire::ErrorKind::Unsupported, describe(reason)), reason_(reason), index_(index) {}
    Reason reason() const noexcept { return reason_; }
    size_t index() const noexcept { return index_; }
    // Static text for a reason; never names scene content.
    static const char* describe(Reason reason) noexcept {
        switch (reason) {
        case Reason::ValueType: return "value type is not representable in version zero";
        case Reason::PayloadOffset: return "payload offset is not representable in version zero";
        case Reason::IdentityTypeChange: return "existing identity type change is not supported";
        case Reason::MultiUpdate: return "structural edit would need several updates";
        case Reason::MoveCycle: return "namespace move cycle would need temporary creates";
        case Reason::TimeSampleDictionaryKey: return "time-sample map edit by dictionary key is not supported";
        case Reason::IdentityRealization: return "namespace change cannot be realized by identity-preserving edits";
        }
        return "unsupported edit";
    }
private:
    Reason reason_;
    size_t index_;
};
} // namespace omni::usd::live
