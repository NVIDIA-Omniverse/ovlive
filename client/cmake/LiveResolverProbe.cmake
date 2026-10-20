# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Test-only USD hook-order probe (M0.4); never installed or shipped.
if(NOT TARGET live_scene_dependencies OR NOT TARGET live_uri)
    message(FATAL_ERROR "LIVE_BUILD_RESOLVER_PROBE requires LIVE_BUILD_SCENE_CORE and its explicit USD SDK inputs")
endif()
if(NOT BUILD_TESTING OR NOT CMAKE_SYSTEM_NAME MATCHES "^(Linux|Windows|Darwin)$")
    message(FATAL_ERROR "Resolver probe requires BUILD_TESTING and the Linux, Windows or macOS USD 25.11 SDK")
endif()
add_library(live_resolver_probe_dependencies INTERFACE)
target_link_libraries(live_resolver_probe_dependencies INTERFACE live_scene_dependencies)
live_link_usd(live_resolver_probe_dependencies INTERFACE ar plug)
add_library(live_resolver_probe_plugin SHARED tests/native/resolver/Plugin.cpp)
target_include_directories(live_resolver_probe_plugin PRIVATE source/live/runtime)
target_link_libraries(live_resolver_probe_plugin PRIVATE live_uri live_resolver_probe_dependencies)
target_compile_features(live_resolver_probe_plugin PRIVATE cxx_std_17)
# The probe executable calls the plugin's recording functions (Probe.h) directly.
set_target_properties(live_resolver_probe_plugin PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_executable(live_resolver_probe tests/native/resolver/Probe.cpp)
live_link_stack_dump(live_resolver_probe)
target_link_libraries(live_resolver_probe PRIVATE live_resolver_probe_plugin live_resolver_probe_dependencies)
target_compile_features(live_resolver_probe PRIVATE cxx_std_17)
foreach(target live_resolver_probe_plugin live_resolver_probe)
    # Pinned USD headers include the SDK's legacy hash_set compatibility header;
    # its documented suppression avoids promoting that SDK warning to an error.
    live_warnings(${target} USD TEST)
    live_sanitize(${target} PRIVATE)
endforeach()
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/resolver-probe/plugInfo.json"
    INPUT "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/resolver/plugInfo.json.in")
find_package(Python3 COMPONENTS Interpreter REQUIRED)
file(GLOB_RECURSE resolver_identity_inputs CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/resolver/*")
# Used headers already participate in compiler dependency tracking. Also relink
# the driver when any recorded SDK input changes, so its post-build identity
# cannot silently remain stale when the SDK's library/configuration changes.
file(GLOB_RECURSE resolver_sdk_inputs CONFIGURE_DEPENDS
    "${LIVE_USD_ROOT}/include/*" "${LIVE_USD_ROOT}/lib/*"
    "${LIVE_USD_PYTHON_ROOT}/include/*" "${LIVE_USD_PYTHON_ROOT}/lib/*")
set_property(TARGET live_resolver_probe APPEND PROPERTY LINK_DEPENDS ${resolver_identity_inputs}
    ${resolver_sdk_inputs} "$<TARGET_FILE:live_resolver_probe_plugin>"
    "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveResolverProbe.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveSceneCore.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveUsdSdk.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveSanitizer.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/resolver_identity.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/sdk_tree_identity.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/test.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/omni/usd/live/Live.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/LiveUri.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/LiveUri.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/LiveUriApi.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/resolver-probe/plugInfo.json")
add_custom_command(TARGET live_resolver_probe POST_BUILD
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/resolver_identity.py"
        --root "${CMAKE_CURRENT_SOURCE_DIR}" --artifact "$<TARGET_FILE:live_resolver_probe>"
        --plugin "$<TARGET_FILE:live_resolver_probe_plugin>"
        --plug-info "${CMAKE_CURRENT_BINARY_DIR}/resolver-probe/plugInfo.json"
        --usd "${LIVE_USD_ROOT}" --python "${LIVE_USD_PYTHON_ROOT}"
        --compiler-id "${CMAKE_CXX_COMPILER_ID}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
        --configuration "$<CONFIG>" ${LIVE_SANITIZER_MANIFEST_ARGS}
    VERBATIM)
add_test(NAME live_resolver_hook_probe COMMAND live_resolver_probe
    "${CMAKE_CURRENT_BINARY_DIR}/resolver-probe/plugInfo.json"
    "${CMAKE_CURRENT_BINARY_DIR}/resolver-probe/trace.json")
set_tests_properties(live_resolver_hook_probe PROPERTIES TIMEOUT 30
    ENVIRONMENT "PXR_PLUGINPATH_NAME=;PXR_AR_DEFAULT_SEARCH_PATH=;AR_DEFAULT_RESOLVER=")
add_test(NAME live_creation_hook_contract COMMAND ${Python3_EXECUTABLE} -B
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/resolver/creation.py"
    "$<TARGET_FILE:live_resolver_probe>"
    "${CMAKE_CURRENT_BINARY_DIR}/resolver-probe/plugInfo.json"
    "${CMAKE_CURRENT_BINARY_DIR}/resolver-probe/creation")
set_tests_properties(live_creation_hook_contract PROPERTIES TIMEOUT 100
    ENVIRONMENT "PXR_PLUGINPATH_NAME=;PXR_AR_DEFAULT_SEARCH_PATH=;AR_DEFAULT_RESOLVER=;PYTHONDONTWRITEBYTECODE=1")
