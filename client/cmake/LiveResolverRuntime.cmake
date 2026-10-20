# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
if(NOT TARGET live_runtime OR NOT TARGET live_scene_dependencies)
    message(FATAL_ERROR "LIVE_BUILD_RESOLVER_RUNTIME requires explicit native runtime and scene SDK inputs")
endif()
# A component build of the production resolver source for isolated tests; the
# installable omni_usd_live plugin (LivePlugin.cmake) compiles the same source.
add_library(live_resolver_runtime_dependencies INTERFACE)
target_link_libraries(live_resolver_runtime_dependencies INTERFACE live_scene_dependencies)
live_link_usd(live_resolver_runtime_dependencies INTERFACE ar plug)
add_library(live_resolver_runtime_component SHARED source/live/resolver/LiveResolver.cpp)
target_link_libraries(live_resolver_runtime_component PRIVATE live_runtime live_uri live_resolver_runtime_dependencies)
target_compile_features(live_resolver_runtime_component PRIVATE cxx_std_17)
live_warnings(live_resolver_runtime_component USD)
set_target_properties(live_resolver_runtime_component PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(UNIX AND NOT APPLE)
    # USD registration uses static factory registration; it needs no exported
    # C++ symbols. Keep archive/STL symbols private when loaded beside Kit.
    target_link_options(live_resolver_runtime_component PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/Live.exports")
    set_property(TARGET live_resolver_runtime_component APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/Live.exports")
endif()
set(OMNI_USD_LIVE_RESOLVER_LIBRARY "$<TARGET_FILE:live_resolver_runtime_component>")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/source/live/resolver/plugInfo.json.in"
    "${CMAKE_CURRENT_BINARY_DIR}/resolver-runtime/plugInfo.json.gen" @ONLY NEWLINE_STYLE UNIX)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/resolver-runtime/plugInfo.json"
    INPUT "${CMAKE_CURRENT_BINARY_DIR}/resolver-runtime/plugInfo.json.gen")
live_sanitize(live_resolver_runtime_component PRIVATE)
if(BUILD_TESTING)
    add_executable(live_resolver_runtime_driver tests/native/resolver_runtime/driver.cpp)
    live_link_stack_dump(live_resolver_runtime_driver)
    target_include_directories(live_resolver_runtime_driver PRIVATE include)
    target_compile_features(live_resolver_runtime_driver PRIVATE cxx_std_17)
    # Only the shared component owns runtime objects: the public API and the Ar
    # factory must observe the same endpoint registry and shutdown state.
    target_link_libraries(live_resolver_runtime_driver PRIVATE live_resolver_runtime_component live_resolver_runtime_dependencies)
    live_warnings(live_resolver_runtime_driver USD TEST)
    live_sanitize(live_resolver_runtime_driver PRIVATE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    get_property(resolver_dependencies GLOBAL PROPERTY LIVE_DEPENDENCY_FILES)
    set(resolver_dependency_args)
    foreach(dependency IN LISTS resolver_dependencies)
        list(APPEND resolver_dependency_args --dependency "${dependency}")
    endforeach()
    file(GLOB_RECURSE resolver_runtime_inputs CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/resolver/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/protos/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/resolver_runtime/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/reference_service/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/harness/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/*")
    list(FILTER resolver_runtime_inputs EXCLUDE REGEX "/__pycache__/|\\.py[co]$")
    file(GLOB_RECURSE resolver_runtime_sdk_inputs CONFIGURE_DEPENDS
        "${LIVE_USD_ROOT}/include/*" "${LIVE_USD_ROOT}/lib/*"
        "${LIVE_USD_PYTHON_ROOT}/include/*" "${LIVE_USD_PYTHON_ROOT}/lib/*")
    set_property(TARGET live_resolver_runtime_driver APPEND PROPERTY LINK_DEPENDS
        ${resolver_runtime_inputs} ${resolver_runtime_sdk_inputs} ${resolver_dependencies}
        "$<TARGET_FILE:live_resolver_runtime_component>"
        "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveSceneCore.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveUsdSdk.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveSanitizer.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveGrpcDependencies.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveBuildProvenance.h.in"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveResolverRuntime.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/omni/usd/live/Live.h"
        "${CMAKE_CURRENT_BINARY_DIR}/resolver-runtime/plugInfo.json")
    add_custom_command(TARGET live_resolver_runtime_driver POST_BUILD
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/resolver_runtime_identity.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}" --artifact "$<TARGET_FILE:live_resolver_runtime_driver>"
            --plugin "$<TARGET_FILE:live_resolver_runtime_component>"
            --plug-info "${CMAKE_CURRENT_BINARY_DIR}/resolver-runtime/plugInfo.json"
            --usd "${LIVE_USD_ROOT}" --python "${LIVE_USD_PYTHON_ROOT}"
            --compiler-id "${CMAKE_CXX_COMPILER_ID}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
            --configuration "$<CONFIG>" ${LIVE_SANITIZER_MANIFEST_ARGS}
            ${resolver_dependency_args}
        VERBATIM)
endif()
