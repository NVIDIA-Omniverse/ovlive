# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The FlatBuffers runtime headers the generated OMPF schema requires are
# vendored byte-exact (source/live/wire/third_party/flatbuffers/README.md); the
# build takes no external FlatBuffers root. Later modules read this variable.
unset(LIVE_FLATBUFFERS_ROOT CACHE)
set(LIVE_FLATBUFFERS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/source/live/wire/third_party/flatbuffers")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/source/live/wire/schema/provenance.json" wire_provenance)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/source/live/wire/schema/provenance.json")
foreach(header flatbuffers.h base.h stl_emulation.h)
    set(path "${LIVE_FLATBUFFERS_ROOT}/include/flatbuffers/${header}")
    string(JSON expected_hash GET "${wire_provenance}" flatbuffers headers "${header}")
    file(SHA256 "${path}" actual_hash)
    if(NOT actual_hash STREQUAL expected_hash)
        message(FATAL_ERROR "FlatBuffers ${header} differs from the wire-schema provenance pin")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
endforeach()
add_library(live_wire STATIC source/live/wire/OmpfEnvelope.cpp source/live/wire/OmpfEncode.cpp
    source/live/wire/OmpfHash.cpp source/live/support/SHA256.cpp)
target_include_directories(live_wire PUBLIC source/live/wire source/live/support PRIVATE include)
target_include_directories(live_wire SYSTEM PRIVATE "${LIVE_FLATBUFFERS_ROOT}/include")
target_compile_features(live_wire PUBLIC cxx_std_17)
set_target_properties(live_wire PROPERTIES POSITION_INDEPENDENT_CODE ON)
live_sanitize(live_wire PUBLIC)
if(BUILD_TESTING)
    add_executable(live_wire_envelope_test tests/native/wire/envelope.cpp tests/native/wire/envelope_encode.cpp)
    live_link_stack_dump(live_wire_envelope_test)
    target_link_libraries(live_wire_envelope_test PRIVATE live_wire)
    target_include_directories(live_wire_envelope_test SYSTEM PRIVATE "${LIVE_FLATBUFFERS_ROOT}/include")
    target_compile_options(live_wire_envelope_test PRIVATE -UNDEBUG)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    file(GLOB_RECURSE wire_identity_inputs CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/wire/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/wire/*"
        "${LIVE_FLATBUFFERS_ROOT}/include/*")
    set_property(TARGET live_wire_envelope_test APPEND PROPERTY LINK_DEPENDS ${wire_identity_inputs}
        "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveWire.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/support/SHA256.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/support/SHA256.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/wire_identity.py" "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/test.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/sdk_tree_identity.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/test_support.py" "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/process_guard.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/sanitizer.py")
    add_custom_command(TARGET live_wire_envelope_test POST_BUILD
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/wire_identity.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}" --artifact "$<TARGET_FILE:live_wire_envelope_test>"
            --flatbuffers "${LIVE_FLATBUFFERS_ROOT}"
            --compiler-id "${CMAKE_CXX_COMPILER_ID}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
            --configuration "$<CONFIG>" ${LIVE_SANITIZER_MANIFEST_ARGS}
        VERBATIM)
    add_test(NAME live_wire_envelope COMMAND live_wire_envelope_test 42)
    set_tests_properties(live_wire_envelope PROPERTIES TIMEOUT 60)
endif()
