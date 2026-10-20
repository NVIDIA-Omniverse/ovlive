// SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "OmniObjectDelta_generated.h"

struct OmniObjectDeltaBuilder
{
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<uint8_t> structCommandTypes;
    std::vector<flatbuffers::Offset<void>> structCommandOffsets;
    std::vector<flatbuffers::Offset<OmniObjectDelta::SetField>> setFieldOffsets;
    std::vector<flatbuffers::Offset<OmniObjectDelta::TimeSample>> timeSampleOffsets;

    void Clear()
    {
        fbb.Reset();
        structCommandTypes.clear();
        structCommandOffsets.clear();
        setFieldOffsets.clear();
        timeSampleOffsets.clear();
    }

    template<typename T>
    void AddStructCommand(const flatbuffers::Offset<T> & offset)
    {
        structCommandTypes.push_back(OmniObjectDelta::StructCommandTraits<T>::enum_value);
        structCommandOffsets.push_back(offset.Union());
    }

    void AddSetField(const flatbuffers::Offset<OmniObjectDelta::SetField> & offset)
    {
        setFieldOffsets.push_back(offset);
    }

    void AddTimeSample(const flatbuffers::Offset<OmniObjectDelta::TimeSample>& offset)
    {
        timeSampleOffsets.push_back(offset);
    }

    void Finish(bool isDiff, uint64_t diffBaseVersion, const char* sourceFormat = nullptr)
    {
        auto delta = OmniObjectDelta::CreateDeltaDirect(fbb, isDiff, diffBaseVersion, &structCommandTypes, &structCommandOffsets, &setFieldOffsets, &timeSampleOffsets, sourceFormat);
        OmniObjectDelta::FinishDeltaBuffer(fbb, delta);
    }
};
