// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pxr/base/tf/token.h>
#include <pxr/base/vt/value.h>
#include <pxr/usd/sdf/types.h>

#include <deque>
#include <memory>
#include <string>

// forwards
struct Spec;
struct SpecFields;

// current and committed.
struct SpecField final {
private:
    struct Entry {
        Entry(PXR_NS::VtValue value, const uint64_t version) : value(std::move(value)), version(version) {}
        Entry(const Entry& other) = delete;
        Entry(Entry&& other) noexcept : value(std::move(other.value)), version(other.version) {}

        ~Entry() = default;

        Entry& operator=(const Entry& other) = delete;

        Entry& operator=(Entry&& other) noexcept {
            using std::swap;
            swap(value, other.value);
            std::swap(version, other.version);
            return *this;
        }

        PXR_NS::VtValue value;
        uint64_t version;
    };

    using Version_t = std::deque<Entry>;

public:
    SpecField(const std::weak_ptr<Spec>& spec, uint64_t specId, const PXR_NS::TfToken& fieldName);
    SpecField(std::weak_ptr<Spec> spec, uint64_t specId, const PXR_NS::TfToken& fieldName, PXR_NS::VtValue&& value);
    SpecField(const SpecField& other) = delete;
    SpecField(SpecField&& other) = default;
    ~SpecField() = default;

    SpecField& operator=(SpecField& other) = delete;
    SpecField& operator=(SpecField&& other) = delete;


    [[nodiscard]] const PXR_NS::VtValue& CurrentValue() const;
    [[nodiscard]] std::string GetString(bool currentValueOnly = false) const;

    explicit operator const PXR_NS::VtValue&() const;
    bool CommitValue(PXR_NS::VtValue&& value, uint64_t version);
    void CommitVersion(uint64_t version);
    [[nodiscard]] Version_t::value_type& Current();
    [[nodiscard]] uint64_t CurrentVersion() const;
    [[nodiscard]] const PXR_NS::VtValue& CommittedValue() const;
    [[nodiscard]] uint64_t CommittedVersion() const;
    void EraseTimeSample(const double time, const uint64_t version);
    void InsertOrAssignTimeSample(const double time, const PXR_NS::VtValue& value, const uint64_t version);
    static bool IsEmpty(const PXR_NS::VtValue& value);
    void SetValue(PXR_NS::VtValue&& otherValue);
    // Exact scene storage: one value, no inferred local/remote acknowledgment.
    // Construct before publication so allocation failure leaves this unchanged.
    void ReplaceValue(PXR_NS::VtValue value);

    // Populated by debug builds. Keep the API and object layout independent of
    // assertion settings: assertion-enabled callers also link Release libraries.
    [[nodiscard]] const std::string& FieldPath() const { return _fieldPath; }

    static std::string VtValueToStr(const PXR_NS::VtValue& value);

    friend void swap(SpecField& a, SpecField& b) noexcept;

private:
    std::weak_ptr<Spec> _spec;
    Version_t _versions;

    std::string _fieldPath;

public:
    static const PXR_NS::VtValue DefaultValue;
};
