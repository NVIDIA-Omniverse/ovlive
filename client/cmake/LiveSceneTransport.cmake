# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
if(NOT TARGET live_runtime OR NOT TARGET live_scene_journal)
    message(FATAL_ERROR "LIVE_BUILD_SCENE_TRANSPORT requires LIVE_BUILD_GRPC_RUNTIME and LIVE_BUILD_WIRE_SCENE")
endif()
add_library(live_scene_transport STATIC source/live/scene/SceneAttachment.cpp source/live/scene/SceneCreation.cpp
    source/live/scene/UsdLayerBinding.cpp source/live/scene/LiveSdfFileFormat.cpp source/live/scene/UsdLayerCreation.cpp
    source/live/scene/UsdCreationRegistry.cpp)
target_include_directories(live_scene_transport PUBLIC source/live/scene)
# Object-library dependencies do not propagate their objects through an
# interface: retain both runtime and URI objects in this archive explicitly.
target_link_libraries(live_scene_transport PUBLIC live_scene_journal live_runtime live_uri)
target_compile_features(live_scene_transport PUBLIC cxx_std_17)
live_warnings(live_scene_transport USD)
# The transport defines C ABI functions that the plugin exports.
target_compile_definitions(live_scene_transport PRIVATE OMNIUSDLIVE_EXPORTS)
set_target_properties(live_scene_transport PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(BUILD_TESTING)
    add_executable(live_scene_transport_driver tests/native/scene/scene_transport.cpp tests/native/scene/scene_publication.cpp
        tests/native/scene/scene_file_format.cpp source/live/resolver/LiveResolver.cpp)
    live_link_stack_dump(live_scene_transport_driver)
    target_link_libraries(live_scene_transport_driver PRIVATE live_scene_transport)
    live_link_usd(live_scene_transport_driver PRIVATE ar plug usd)
    # Already-linked production types share this driver's one runtime. The
    # installed shared-plugin/public-host boundary is qualified separately.
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/source/live/scene/plugInfo.json.in"
        "${CMAKE_CURRENT_BINARY_DIR}/native-file-format/plugInfo.json" COPYONLY)
    target_compile_definitions(live_scene_transport_driver PRIVATE OMNIUSDLIVE_EXPORTS)
    target_compile_options(live_scene_transport_driver PRIVATE -UNDEBUG)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    get_property(scene_transport_dependencies GLOBAL PROPERTY LIVE_DEPENDENCY_FILES)
    set(scene_transport_dependency_args)
    foreach(dependency IN LISTS scene_transport_dependencies)
        list(APPEND scene_transport_dependency_args --dependency "${dependency}")
    endforeach()
    file(GLOB_RECURSE scene_transport_identity_inputs CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/include/omni/usd/live/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/scene/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/wire/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/resolver/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/protos/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/support/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/scene/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/wire/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/runtime/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/reference_service/*.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/*.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/*.sh"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/*.bat"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/harness/*.py"
        "${LIVE_FLATBUFFERS_ROOT}/include/*"
        "${LIVE_USD_ROOT}/include/*" "${LIVE_USD_ROOT}/lib/*"
        "${LIVE_USD_PYTHON_ROOT}/include/*" "${LIVE_USD_PYTHON_ROOT}/lib/*")
    set_property(TARGET live_scene_transport_driver APPEND PROPERTY LINK_DEPENDS
        ${scene_transport_identity_inputs} ${scene_transport_dependencies}
        "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveSceneTransport.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveGrpcDependencies.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveBuildProvenance.h.in"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveWire.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveUsdWire.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveWireScene.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveSceneCore.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveUsdSdk.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveSanitizer.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/omni/usd/live/Live.h")
    set_property(TARGET live_scene_transport_driver APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_BINARY_DIR}/native-file-format/plugInfo.json")
    add_custom_command(TARGET live_scene_transport_driver POST_BUILD
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/wire_identity.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}" --artifact "$<TARGET_FILE:live_scene_transport_driver>"
            --boundary scene_transport --flatbuffers "${LIVE_FLATBUFFERS_ROOT}"
            --usd "${LIVE_USD_ROOT}" --python "${LIVE_USD_PYTHON_ROOT}"
            --compiler-id "${CMAKE_CXX_COMPILER_ID}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
            --configuration "$<CONFIG>" ${LIVE_SANITIZER_MANIFEST_ARGS}
            --dependency "${CMAKE_CURRENT_BINARY_DIR}/native-file-format/plugInfo.json"
            ${scene_transport_dependency_args}
        VERBATIM)
    add_test(NAME live_scene_transport_coverage COMMAND live_scene_transport_driver --coverage)
    set_tests_properties(live_scene_transport_coverage PROPERTIES TIMEOUT 60)
endif()
