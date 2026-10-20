// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "OmpfEnvelope.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace omni::usd::live {
// The Live Edit service's entry limits for one update or creation seed (design
// section 8, Quotas). The service counts a delta's struct commands, set fields
// and time samples together against its MAX_DELTA_ENTRIES; a Reorder is one
// struct command, and the children list it carries is outside that count and
// has a limit of its own. The defaults are the service's limits; only tests
// lower them.
struct DeltaEntryLimits {
    size_t maxEntries = 99999;
    size_t maxChildrenListEntries = 100000;
};
// A delta over a DeltaEntryLimits limit, before anything is issued. Its counts
// are those the service's DELTA_TOO_LARGE or CHILDREN_LIST_TOO_LONG ErrorInfo
// reports for the same delta. It is ErrorKind::ResourceLimit, so a caller that
// does not handle it reports RESOURCE_EXHAUSTED.
class DeltaEntryLimitExceeded final : public wire::DecodeError {
public:
    // The service's ErrorInfo reasons, numbered as the rpc_failed record's reason.
    enum class Reason : uint32_t { DeltaTooLarge = 1, ChildrenListTooLong = 2 };
    struct Counts {
        uint64_t structCommands = 0, setFields = 0, timeSamples = 0;
        uint64_t limit = 0; // max_entries, or the children list limit.
        uint64_t actual = 0; // All entries, or the first over-limit children list's length.
    };
    DeltaEntryLimitExceeded(Reason reason, Counts counts)
        : wire::DecodeError(wire::ErrorKind::ResourceLimit,
              reason == Reason::DeltaTooLarge ? "delta entry limit exceeded" : "children list limit exceeded"),
          reason_(reason), counts_(counts) {}
    Reason reason() const noexcept { return reason_; }
    const Counts& counts() const noexcept { return counts_; }
    // The reason and counts in the service's ErrorInfo terms, such as
    // "DELTA_TOO_LARGE struct_commands=1 set_fields=2 time_samples=3 max_entries=5".
    std::string describe() const {
        if (reason_ == Reason::ChildrenListTooLong)
            return "CHILDREN_LIST_TOO_LONG limit=" + std::to_string(counts_.limit) +
                " actual=" + std::to_string(counts_.actual);
        return "DELTA_TOO_LARGE struct_commands=" + std::to_string(counts_.structCommands) +
            " set_fields=" + std::to_string(counts_.setFields) +
            " time_samples=" + std::to_string(counts_.timeSamples) + " max_entries=" + std::to_string(counts_.limit);
    }
private:
    Reason reason_;
    Counts counts_;
};
// Throws DeltaEntryLimitExceeded when a delta of `commands`, `setFields` set
// fields and `timeSamples` time samples exceeds `limits`. The entry count is
// checked first, then each Reorder's children list in command order, as the
// service checks them.
inline void checkDeltaEntries(const std::vector<wire::Command>& commands, size_t setFields, size_t timeSamples,
    const DeltaEntryLimits& limits) {
    const uint64_t structCommands = commands.size();
    const uint64_t entries = structCommands + setFields + timeSamples;
    if (entries > limits.maxEntries)
        throw DeltaEntryLimitExceeded(DeltaEntryLimitExceeded::Reason::DeltaTooLarge,
            {structCommands, setFields, timeSamples, limits.maxEntries, entries});
    for (const auto& command : commands)
        if (command.kind == wire::Command::Kind::Reorder && command.children.size() > limits.maxChildrenListEntries)
            throw DeltaEntryLimitExceeded(DeltaEntryLimitExceeded::Reason::ChildrenListTooLong,
                {structCommands, setFields, timeSamples, limits.maxChildrenListEntries, command.children.size()});
}
} // namespace omni::usd::live
