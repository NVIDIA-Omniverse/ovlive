// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// Shared C++ test fixtures for the oodf-ffi test harness. These are available
// to every TU compiled into the test binary (see oodf-ffi/build.rs). The
// fixtures live outside the shipping static library — they exist solely for
// the standalone test binary spawned by oodf-ffi/tests/cpp_tests.rs.

#include <cstddef>
#include <cstdint>

namespace oodf_test {

// 280-byte OODF payload harvested from the legacy C++ live-edit service (the
// "Postman" integration sample). The base64 form is documented in
// src/domain/ompf.rs::parses_cpp_service_ompf_chunk (constant B64_CHUNK). The
// 16-byte OMPF container header has been stripped here — this array is only
// the OODF FlatBuffers payload, ready to hand to `oodf::parse_delta`.
extern const uint8_t kPostmanOodfPayload[];
extern const size_t kPostmanOodfPayloadLen;

}  // namespace oodf_test
