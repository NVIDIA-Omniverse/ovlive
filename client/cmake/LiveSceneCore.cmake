# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Development SDK inputs are explicit. This target never fetches dependencies or
# links omni.client or another live delegate or resolver. Package/platform qualification
# is separate from compiling this actual scene implementation.
set(LIVE_USD_ROOT "" CACHE PATH "USD SDK root (headers and lib)")
set(LIVE_USD_PYTHON_ROOT "" CACHE PATH "Python SDK used by the selected USD build")
foreach(input LIVE_USD_ROOT LIVE_USD_PYTHON_ROOT)
    if(NOT IS_DIRECTORY "${${input}}/include")
        message(FATAL_ERROR "${input} must name an explicit SDK root with an include directory")
    endif()
endforeach()
if(NOT CMAKE_SYSTEM_NAME MATCHES "^(Linux|Windows|Darwin)$")
    message(FATAL_ERROR "Scene core SDK linkage supports Linux, Windows and macOS")
endif()
# The Python 3.12 headers, library and interpreter of the SDK Python: a Windows Python package keeps its headers in
# include, its import library in libs and its interpreter at the root.
if(WIN32)
    set(LIVE_PYTHON_INCLUDE "${LIVE_USD_PYTHON_ROOT}/include")
    set(LIVE_PYTHON_LIBRARY "${LIVE_USD_PYTHON_ROOT}/libs/python312.lib")
    set(LIVE_PYTHON_INTERPRETER "${LIVE_USD_PYTHON_ROOT}/python.exe")
else()
    set(LIVE_PYTHON_INCLUDE "${LIVE_USD_PYTHON_ROOT}/include/python3.12")
    set(LIVE_PYTHON_LIBRARY "${LIVE_USD_PYTHON_ROOT}/lib/libpython3.12${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(LIVE_PYTHON_INTERPRETER "${LIVE_USD_PYTHON_ROOT}/bin/python3.12")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_library(live_scene_dependencies INTERFACE)
target_include_directories(live_scene_dependencies SYSTEM INTERFACE "${LIVE_USD_ROOT}/include" "${LIVE_PYTHON_INCLUDE}")
# Reject missing or mismatched inputs instead of resolving an ambient SDK.
live_link_usd(live_scene_dependencies INTERFACE sdf usdGeom vt gf tf arch trace work python tbb)
if(NOT EXISTS "${LIVE_PYTHON_LIBRARY}")
    message(FATAL_ERROR "The current scene SDK lane requires Python 3.12: ${LIVE_PYTHON_LIBRARY}")
endif()
target_link_libraries(live_scene_dependencies INTERFACE "${LIVE_PYTHON_LIBRARY}")
if(APPLE)
    find_program(live_install_name_tool install_name_tool REQUIRED)
    find_program(live_codesign codesign REQUIRED)
    find_program(live_otool otool REQUIRED)
    execute_process(COMMAND "${live_otool}" -D "${LIVE_PYTHON_LIBRARY}"
        OUTPUT_VARIABLE live_python_install_name OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    string(REGEX REPLACE "^[^\n]*\n" "" live_python_install_name "${live_python_install_name}")
    string(STRIP "${live_python_install_name}" live_python_install_name)
    if(NOT live_python_install_name OR live_python_install_name MATCHES "[\n;]")
        message(FATAL_ERROR "Expected one Python SDK Mach-O install name")
    endif()
endif()

function(live_portable_python target)
    if(APPLE)
        # Package consumers resolve the selected host Python SDK through its lib directory.
        # The framework's machine-specific install name is absent from shipped binaries.
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${live_install_name_tool}" -change "${live_python_install_name}"
                "@rpath/libpython3.12.dylib" "$<TARGET_FILE:${target}>"
            COMMAND "${live_codesign}" --force --sign - "$<TARGET_FILE:${target}>"
            VERBATIM)
    endif()
endfunction()
if(NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(live_scene_dependencies INTERFACE ${CMAKE_DL_LIBS} Threads::Threads)
endif()
add_library(live_scene_core STATIC
    source/live/support/Spec.cpp
    source/live/support/SpecField.cpp
    source/live/support/SHA256.cpp
    source/live/support/ReaderWriterLock.cpp)
target_sources(live_scene_core PRIVATE source/live/scene/SdfSceneData.cpp source/live/scene/AbstractValue.cpp)
target_include_directories(live_scene_core PUBLIC source/live/scene)
target_include_directories(live_scene_core PUBLIC include source/live/support)
target_compile_features(live_scene_core PUBLIC cxx_std_17)
target_link_libraries(live_scene_core PUBLIC live_scene_dependencies)
set_target_properties(live_scene_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
live_sanitize(live_scene_core PUBLIC)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_executable(live_sdf_scene_test tests/native/usd/sdf_scene.cpp)
    live_link_stack_dump(live_sdf_scene_test)
    target_link_libraries(live_sdf_scene_test PRIVATE live_scene_core)
    target_compile_options(live_sdf_scene_test PRIVATE -UNDEBUG)
    add_test(NAME live_sdf_scene COMMAND live_sdf_scene_test 42)
    set_tests_properties(live_sdf_scene PROPERTIES TIMEOUT 60)
    add_executable(live_scene_benchmark tests/native/usd/scene_benchmark.cpp)
    target_link_libraries(live_scene_benchmark PRIVATE live_scene_core)
    # Explicit diagnostic, never a passing product/release test. Its no-op data
    # mode intentionally demonstrates an unsafe SDK boundary in a fresh process.
    add_executable(live_usd_edit_boundary_probe EXCLUDE_FROM_ALL tests/native/usd/edit_boundary_probe.cpp)
    target_link_libraries(live_usd_edit_boundary_probe PRIVATE live_scene_dependencies)
    target_compile_features(live_usd_edit_boundary_probe PRIVATE cxx_std_17)
    set_property(TARGET live_usd_edit_boundary_probe APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/usd_edit_boundary.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/test.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/test_support.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/process_guard.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/sanitizer.py")
    live_sanitize(live_usd_edit_boundary_probe PRIVATE)
    foreach(scene_target live_sdf_scene_test live_scene_benchmark live_usd_edit_boundary_probe)
        if(scene_target STREQUAL "live_scene_benchmark")
            set(scene_boundary "scene_benchmark")
            set(scene_compile_identity --compile-commands "${CMAKE_BINARY_DIR}/compile_commands.json")
            set_property(TARGET ${scene_target} APPEND PROPERTY LINK_DEPENDS "${CMAKE_BINARY_DIR}/compile_commands.json")
        elseif(scene_target STREQUAL "live_usd_edit_boundary_probe")
            set(scene_compile_identity "")
            set(scene_boundary "usd_edit_boundary_probe")
        else()
            set(scene_compile_identity "")
            set(scene_boundary "sdf_scene")
        endif()
        file(GLOB_RECURSE scene_identity_inputs CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/include/omni/usd/live/*"
            "${CMAKE_CURRENT_SOURCE_DIR}/source/live/support/*"
            "${CMAKE_CURRENT_SOURCE_DIR}/source/live/scene/*"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/usd/*")
        set_property(TARGET ${scene_target} APPEND PROPERTY LINK_DEPENDS ${scene_identity_inputs} ${LIVE_TEST_SUPPORT_INPUTS}
            "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveSceneCore.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveUsdSdk.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveSanitizer.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/scene_identity.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/sdk_tree_identity.py")
        add_custom_command(TARGET ${scene_target} POST_BUILD
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/scene_identity.py"
                --root "${CMAKE_CURRENT_SOURCE_DIR}" --artifact "$<TARGET_FILE:${scene_target}>"
                --usd "${LIVE_USD_ROOT}" --python "${LIVE_USD_PYTHON_ROOT}"
                --boundary "${scene_boundary}" ${scene_compile_identity}
                --compiler-id "${CMAKE_CXX_COMPILER_ID}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
                --configuration "$<CONFIG>" ${LIVE_SANITIZER_MANIFEST_ARGS}
            VERBATIM)
    endforeach()
endif()
