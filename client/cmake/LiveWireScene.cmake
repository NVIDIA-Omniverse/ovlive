# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
if(NOT TARGET live_usd_wire OR NOT TARGET live_scene_core)
    message(FATAL_ERROR "LIVE_BUILD_WIRE_SCENE requires LIVE_BUILD_USD_WIRE and LIVE_BUILD_SCENE_CORE")
endif()
add_library(live_wire_scene STATIC source/live/scene/WireScene.cpp source/live/scene/SceneNamespace.cpp source/live/scene/SceneSeed.cpp
    source/live/scene/HostSceneProjection.cpp source/live/scene/HostSceneStorage.cpp source/live/scene/HostLayerState.cpp)
target_include_directories(live_wire_scene PUBLIC source/live/scene)
target_link_libraries(live_wire_scene PUBLIC live_usd_wire live_scene_core)
target_compile_features(live_wire_scene PUBLIC cxx_std_17)
set_target_properties(live_wire_scene PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(live_scene_journal STATIC source/live/scene/SceneJournal.cpp source/live/scene/FieldEditIntent.cpp
    source/live/runtime/LiveLayerProgress.cpp)
target_include_directories(live_scene_journal PUBLIC source/live/scene source/live/runtime)
target_link_libraries(live_scene_journal PUBLIC live_wire_scene live_uri)
live_link_usd(live_scene_journal PUBLIC ar plug)
target_compile_features(live_scene_journal PUBLIC cxx_std_17)
set_target_properties(live_scene_journal PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(BUILD_TESTING)
    add_executable(live_wire_scene_test tests/native/scene/wire_scene.cpp tests/native/scene/host_projection.cpp tests/native/scene/host_projection_alloc.cpp
        tests/native/scene/host_layer_state.cpp)
    live_link_stack_dump(live_wire_scene_test)
    target_include_directories(live_wire_scene_test PRIVATE include)
    target_link_libraries(live_wire_scene_test PRIVATE live_wire_scene)
    live_link_usd(live_wire_scene_test PRIVATE usd)
    target_include_directories(live_wire_scene_test SYSTEM PRIVATE "${LIVE_FLATBUFFERS_ROOT}/include")
    target_compile_options(live_wire_scene_test PRIVATE -UNDEBUG)
    add_executable(live_scene_graph_probe tests/native/scene/graph_probe.cpp)
    live_link_stack_dump(live_scene_graph_probe)
    target_include_directories(live_scene_graph_probe PRIVATE include)
    target_link_libraries(live_scene_graph_probe PRIVATE live_wire_scene)
    target_include_directories(live_scene_graph_probe SYSTEM PRIVATE "${LIVE_FLATBUFFERS_ROOT}/include")
    live_warnings(live_scene_graph_probe USD)

    add_executable(live_scene_journal_test tests/native/scene/scene_journal.cpp tests/native/scene/field_edit_intent.cpp
        tests/native/scene/scene_namespace.cpp tests/native/scene/usd_publication.cpp tests/native/scene/scene_seed.cpp
        tests/native/scene/footprint_work.cpp)
    live_link_stack_dump(live_scene_journal_test)
    target_link_libraries(live_scene_journal_test PRIVATE live_scene_journal)
    live_link_usd(live_scene_journal_test PRIVATE usd)
    target_compile_options(live_scene_journal_test PRIVATE -UNDEBUG)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    file(GLOB_RECURSE wire_scene_identity_inputs CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/include/omni/usd/live/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/scene/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/scene/*"
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
    foreach(scene_target live_wire_scene_test live_scene_graph_probe live_scene_journal_test)
        if(NOT scene_target STREQUAL "live_wire_scene_test" AND NOT scene_target STREQUAL "live_scene_graph_probe")
            set(scene_boundary scene_journal)
            set(scene_runtime_identity_inputs
                "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/LiveLayerProgress.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/LiveLayerProgress.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/LiveUri.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/LiveUri.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/LiveUriApi.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/omni/usd/live/Live.h")
        else()
            set(scene_boundary wire_scene)
            set(scene_runtime_identity_inputs)
        endif()
        set_property(TARGET ${scene_target} APPEND PROPERTY LINK_DEPENDS ${wire_scene_identity_inputs} ${scene_runtime_identity_inputs}
            "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveWire.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveUsdWire.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveWireScene.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveSceneCore.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveUsdSdk.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveSanitizer.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/source/live/support/SHA256.cpp")
        add_custom_command(TARGET ${scene_target} POST_BUILD
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/wire_identity.py"
                --root "${CMAKE_CURRENT_SOURCE_DIR}" --artifact "$<TARGET_FILE:${scene_target}>"
                --boundary ${scene_boundary} --flatbuffers "${LIVE_FLATBUFFERS_ROOT}"
                --usd "${LIVE_USD_ROOT}" --python "${LIVE_USD_PYTHON_ROOT}"
                --compiler-id "${CMAKE_CXX_COMPILER_ID}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
                --configuration "$<CONFIG>" ${LIVE_SANITIZER_MANIFEST_ARGS}
            VERBATIM)
    endforeach()
    add_test(NAME live_wire_scene COMMAND live_wire_scene_test 42)
    set_tests_properties(live_wire_scene PROPERTIES TIMEOUT 60)
    add_test(NAME live_scene_graph_reference COMMAND ${Python3_EXECUTABLE} -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/scene/graph_reference.py"
        --probe "$<TARGET_FILE:live_scene_graph_probe>" --seed 42
        --artifact-root "${CMAKE_CURRENT_BINARY_DIR}/graph-reference-artifacts")
    set_tests_properties(live_scene_graph_reference PROPERTIES TIMEOUT 60)
    add_test(NAME live_scene_graph_reduction COMMAND ${Python3_EXECUTABLE} -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/scene/graph_reduce_selftest.py"
        --probe "$<TARGET_FILE:live_scene_graph_probe>"
        --artifact-root "${CMAKE_CURRENT_BINARY_DIR}/graph-reduction-artifacts")
    set_tests_properties(live_scene_graph_reduction PROPERTIES TIMEOUT 60)
    add_test(NAME live_scene_journal COMMAND live_scene_journal_test)
    set_tests_properties(live_scene_journal PROPERTIES TIMEOUT 240)
endif()
