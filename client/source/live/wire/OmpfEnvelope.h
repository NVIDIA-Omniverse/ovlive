// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace omni::usd::live::wire {
enum class ErrorKind { Malformed, Unsupported, ResourceLimit, Integrity };
class DecodeError : public std::runtime_error {
public:
    DecodeError(ErrorKind kind, const char* message) : std::runtime_error(message), kind(kind) {}
    const ErrorKind kind;
};
// Defaults sit above anything admission lets in: a read update is bounded by
// max_pending_bytes_per_endpoint and a sent one by max_update_bytes. Decoding is
// linear in its input except for values shared by several references.
struct EnvelopeLimits {
    size_t maxBytes = size_t(4) << 30;
    uint32_t maxParts = 1u << 20;
    uint32_t maxRecords = 1u << 28; // Commands + fields + samples + reordered IDs.
    size_t maxStringBytes = size_t(4) << 30; // Aggregate copied names, including aliases.
    size_t maxReferencedValueBytes = size_t(16) << 30; // Count each reference, even when shared.
};
class EnvelopeStorage;
class ValueBytes {
public:
    bool empty() const { return size_ == 0; }
    size_t size() const { return size_; }
    std::string_view bytes() const;
private:
    friend class EnvelopeParser;
    std::shared_ptr<const EnvelopeStorage> storage_;
    size_t offset_ = 0, size_ = 0;
};
struct Command {
    enum class Kind { Create, Delete, Move, Reorder, DiffChange, DiffDelete };
    Kind kind;
    uint64_t section = 0, parent = 0, oldParent = 0;
    uint8_t sectionType = 0, childList = 0;
    uint32_t sectionOrder = 0;
    std::string name;
    std::vector<uint64_t> children;
};
struct Field {
    uint64_t section = 0, setOrder = 0;
    std::string name;
    ValueBytes value;
};
struct Sample {
    uint64_t section = 0;
    double time = 0;
    ValueBytes value;
};
// Framing/command/hash validation only. Values are immutable owned byte spans,
// NOT decoded USD values or proof that this batch may advance scene coverage.
struct Envelope {
    bool isDiff = false;
    uint64_t baseVersion = 0;
    std::string sourceFormat;
    std::vector<Command> commands;
    std::vector<Field> fields;
    std::vector<Sample> samples;
};
Envelope decodeEnvelope(std::string_view bytes, const EnvelopeLimits& limits = {});
} // namespace omni::usd::live::wire
