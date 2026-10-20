# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
if(NOT TARGET live_wire OR NOT TARGET live_scene_dependencies)
    message(FATAL_ERROR "LIVE_BUILD_USD_WIRE requires LIVE_BUILD_WIRE and LIVE_BUILD_SCENE_CORE")
endif()
add_library(live_usd_wire STATIC source/live/wire/UsdValueCodec.cpp source/live/wire/UsdEnvelopeEncode.cpp
    source/live/wire/UsdValueNumeric.cpp source/live/wire/UsdValueCompound.cpp
    source/live/wire/UsdValueEncode.cpp source/live/wire/UsdValueWriteNumeric.cpp source/live/wire/UsdValueWriteCompound.cpp)
target_include_directories(live_usd_wire PUBLIC source/live/wire PRIVATE include)
target_link_libraries(live_usd_wire PUBLIC live_wire live_scene_dependencies)
target_compile_features(live_usd_wire PUBLIC cxx_std_17)
set_target_properties(live_usd_wire PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(BUILD_TESTING)
    add_executable(live_usd_values_test tests/native/wire/usd_values.cpp tests/native/wire/usd_envelope_encode.cpp)
    live_link_stack_dump(live_usd_values_test)
    target_include_directories(live_usd_values_test PRIVATE include)
    target_link_libraries(live_usd_values_test PRIVATE live_usd_wire)
    target_include_directories(live_usd_values_test SYSTEM PRIVATE "${LIVE_FLATBUFFERS_ROOT}/include")
    target_compile_options(live_usd_values_test PRIVATE -UNDEBUG)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    file(GLOB_RECURSE usd_values_identity_inputs CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/include/omni/usd/live/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/wire/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/wire/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/support/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/*.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/*.sh"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/*.bat"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/harness/*.py"
        "${LIVE_FLATBUFFERS_ROOT}/include/*"
        "${LIVE_USD_ROOT}/include/*" "${LIVE_USD_ROOT}/lib/*"
        "${LIVE_USD_PYTHON_ROOT}/include/*" "${LIVE_USD_PYTHON_ROOT}/lib/*")
    set_property(TARGET live_usd_values_test APPEND PROPERTY LINK_DEPENDS ${usd_values_identity_inputs}
        "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveWire.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveUsdWire.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveSceneCore.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveUsdSdk.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveSanitizer.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/support/SHA256.cpp")
    add_custom_command(TARGET live_usd_values_test POST_BUILD
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/wire_identity.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}" --artifact "$<TARGET_FILE:live_usd_values_test>"
            --boundary usd_values --flatbuffers "${LIVE_FLATBUFFERS_ROOT}"
            --usd "${LIVE_USD_ROOT}" --python "${LIVE_USD_PYTHON_ROOT}"
            --compiler-id "${CMAKE_CXX_COMPILER_ID}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
            --configuration "$<CONFIG>" ${LIVE_SANITIZER_MANIFEST_ARGS}
        VERBATIM)
    add_test(NAME live_usd_values COMMAND live_usd_values_test 42)
    set_tests_properties(live_usd_values PROPERTIES TIMEOUT 60)
endif()
