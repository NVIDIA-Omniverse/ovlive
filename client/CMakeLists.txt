# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(OmniUsdLive LANGUAGES C CXX)
include(cmake/LiveUsdSdk.cmake)
include(cmake/LiveBuildPaths.cmake)
include(cmake/LiveWarnings.cmake)
if(WIN32)
    # windows.h, which USD's headers include, must not pull in the original winsock.h ahead of gRPC's winsock2.h, nor
    # define min and max; the CRT's portable functions are used as on every other platform.
    add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
    # MSVC 14.38 reports a condition that is constant in one template instantiation (C4127); MSVC 14.50, GCC and
    # Clang do not.
    add_compile_options(/wd4127)
    # Template-heavy USD translation units exceed the default COFF section limit.
    add_compile_options(/bigobj)
    # Executables reserve the 8 MiB stack a Linux main thread has: a multi-mode test driver keeps every mode's locals
    # in one frame, which exceeds the 1 MiB Windows default.
    add_link_options("$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:/STACK:8388608>")
endif()
option(LIVE_ENABLE_SANITIZERS "Alias for LIVE_SANITIZER=address" OFF)
option(LIVE_BUILD_SCENE_CORE "Build exact Sdf scene storage against an explicit USD SDK" OFF)

# This explicit source target will be linked into omni_usd_live with the USD and
# transport runtime. The isolated contract probe below is test-only, not a plugin
# substitute and never installed as a production library.
add_library(live_uri OBJECT source/live/runtime/LiveUri.cpp source/live/runtime/LiveUriApi.cpp)
target_include_directories(live_uri PUBLIC include)
target_compile_features(live_uri PUBLIC cxx_std_17)
target_compile_definitions(live_uri PRIVATE OMNIUSDLIVE_EXPORTS)
set_target_properties(live_uri PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden
                      VISIBILITY_INLINES_HIDDEN ON)
if(MSVC)
    target_compile_options(live_uri PRIVATE /W4 /WX)
    target_link_libraries(live_uri PUBLIC ws2_32)
else()
    target_compile_options(live_uri PRIVATE -Wall -Wextra -Werror -pedantic)
endif()

include(CTest)
include(cmake/LiveSanitizer.cmake)
include(cmake/LiveTestSupport.cmake)
include(cmake/LiveLockContract.cmake)
option(LIVE_BUILD_WIRE "Build bounded OMPF wire decoding with the vendored FlatBuffers headers" OFF)
if(LIVE_BUILD_WIRE)
    include(cmake/LiveWire.cmake)
endif()
if(LIVE_BUILD_SCENE_CORE)
    include(cmake/LiveSceneCore.cmake)
endif()
option(LIVE_BUILD_RESOLVER_PROBE "Probe actual USD resolver/file-format hook order" OFF)
if(LIVE_BUILD_RESOLVER_PROBE)
    include(cmake/LiveResolverProbe.cmake)
endif()
option(LIVE_BUILD_USD_WIRE "Build bounded USD value decoding using the explicit scene SDK" OFF)
if(LIVE_BUILD_USD_WIRE)
    include(cmake/LiveUsdWire.cmake)
endif()
option(LIVE_BUILD_WIRE_SCENE "Build transport-free atomic wire-to-scene application" OFF)
if(LIVE_BUILD_WIRE_SCENE)
    include(cmake/LiveWireScene.cmake)
endif()
if(BUILD_TESTING)
    add_library(live_uri_contract_probe SHARED)
    target_link_libraries(live_uri_contract_probe PRIVATE live_uri)
    if(UNIX AND NOT APPLE)
        target_link_options(live_uri_contract_probe PRIVATE
            "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/Live.exports")
        set_property(TARGET live_uri_contract_probe APPEND PROPERTY LINK_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/Live.exports")
    endif()
    set_property(TARGET live_uri_contract_probe APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/write_build_manifest.py")
    add_executable(live_uri_sanitize tests/native/runtime/uri_sanitize.cpp)
    live_link_stack_dump(live_uri_sanitize)
    target_link_libraries(live_uri_sanitize PRIVATE live_uri_contract_probe)
    target_include_directories(live_uri_sanitize PRIVATE include)
    target_compile_features(live_uri_sanitize PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_options(live_uri_sanitize PRIVATE /W4 /WX /UNDEBUG /wd4389) # C4389: see LiveWarnings.cmake
    else()
        target_compile_options(live_uri_sanitize PRIVATE -Wall -Wextra -Werror -pedantic -UNDEBUG)
    endif()
    add_test(NAME live_uri_native_bounds COMMAND live_uri_sanitize 42)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_custom_command(TARGET live_uri_contract_probe POST_BUILD
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/write_build_manifest.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --artifact "$<TARGET_FILE:live_uri_contract_probe>"
            --compiler-id "${CMAKE_CXX_COMPILER_ID}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
            --configuration "$<CONFIG>" ${LIVE_SANITIZER_MANIFEST_ARGS}
            --input CMakeLists.txt --input include/omni/usd/live/Live.h
            --input source/live/runtime/LiveUri.h --input source/live/runtime/LiveUri.cpp
            --input source/live/runtime/LiveUriApi.cpp --input source/live/runtime/Live.exports
        VERBATIM)

    add_test(NAME live_uri_contract COMMAND ${CMAKE_COMMAND} -E env
        "LIVE_TEST_URI_LIBRARY=$<TARGET_FILE:live_uri_contract_probe>"
        ${Python3_EXECUTABLE} -m unittest discover -s tests/runtime -p test_uri.py -v)
    set_tests_properties(live_uri_contract PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

foreach(target live_uri live_uri_contract_probe live_uri_sanitize)
    live_sanitize(${target} PRIVATE)
endforeach()
if(LIVE_SANITIZED)
    # Python itself is not a sanitizer binary. Run the native driver in this lane;
    # the normal build runs Python's independent semantic oracle via the ABI.
    set_tests_properties(live_uri_contract PROPERTIES DISABLED TRUE)
endif()

option(LIVE_BUILD_GRPC_RUNTIME "Build the native LiveService runtime with pinned gRPC dependencies" OFF)
if(LIVE_BUILD_GRPC_RUNTIME)
    include(cmake/LiveGrpcDependencies.cmake)
    set(live_proto_root "${CMAKE_CURRENT_SOURCE_DIR}/source/live/protos")
    set(live_proto_name "nvidia/omniverse/live/v1alpha/live")
    set(live_generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${live_generated}")
    add_custom_command(
        OUTPUT "${live_generated}/${live_proto_name}.pb.cc" "${live_generated}/${live_proto_name}.pb.h"
               "${live_generated}/${live_proto_name}.grpc.pb.cc" "${live_generated}/${live_proto_name}.grpc.pb.h"
        COMMAND "${LIVE_PROTOC}" "--proto_path=${live_proto_root}"
            "--cpp_out=${live_generated}" "--grpc_out=${live_generated}"
            "--plugin=protoc-gen-grpc=${LIVE_GRPC_CPP_PLUGIN}" "${live_proto_root}/${live_proto_name}.proto"
        DEPENDS "${live_proto_root}/${live_proto_name}.proto" "${LIVE_PROTOC}" "${LIVE_GRPC_CPP_PLUGIN}"
        VERBATIM)
    add_library(live_proto STATIC "${live_generated}/${live_proto_name}.pb.cc"
                                  "${live_generated}/${live_proto_name}.grpc.pb.cc")
    target_include_directories(live_proto PUBLIC "${live_generated}")
    target_link_libraries(live_proto PUBLIC live_grpc_dependencies)
    target_compile_features(live_proto PUBLIC cxx_std_17)
    set_target_properties(live_proto PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden
                          VISIBILITY_INLINES_HIDDEN ON)
    add_library(live_runtime OBJECT source/live/runtime/LiveRuntime.cpp source/live/runtime/LiveRuntimeApi.cpp
                                   source/live/runtime/LiveRequest.cpp source/live/runtime/LiveUnary.cpp
                                   source/live/runtime/LiveDispatch.cpp source/live/runtime/LiveAuth.cpp
                                   source/live/runtime/LiveFinite.cpp source/live/runtime/LiveLayerWrite.cpp
                                   source/live/runtime/LiveLayerRead.cpp source/live/runtime/LiveLayerProgress.cpp source/live/runtime/LiveLayerActor.cpp
                                   source/live/runtime/LiveJitter.cpp
                                   source/live/runtime/LiveDrain.cpp
                                   source/live/runtime/LiveStatus.cpp
                                   source/live/runtime/LiveDiagnostics.cpp source/live/runtime/LiveDiagnosticsApi.cpp
                                   source/live/runtime/LiveSession.cpp source/live/runtime/LiveSessionPeers.cpp
                                   source/live/runtime/LiveSessionApi.cpp source/live/runtime/LiveLayerAdmission.cpp source/live/runtime/LiveLayerStatus.cpp)
    # Whitelisted provenance only; dependency manifests can contain credentials.
    file(SHA256 "${live_proto_root}/${live_proto_name}.proto" live_proto_sha256)
    execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE live_revision OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT live_revision MATCHES "^[0-9a-f]+$")
        set(live_revision "unknown")
    endif()
    file(GLOB live_identity_inputs CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/*"
         "${CMAKE_CURRENT_SOURCE_DIR}/include/omni/usd/live/*")
    list(APPEND live_identity_inputs "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
         "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveGrpcDependencies.cmake"
         "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveBuildProvenance.h.in"
         "${live_proto_root}/${live_proto_name}.proto")
    list(SORT live_identity_inputs)
    set(live_identity_text "")
    foreach(input IN LISTS live_identity_inputs)
        file(SHA256 "${input}" digest)
        file(RELATIVE_PATH relative "${CMAKE_CURRENT_SOURCE_DIR}" "${input}")
        string(APPEND live_identity_text "${relative}:${digest}\n")
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${live_identity_inputs})
    string(SHA256 live_source_sha256 "${live_identity_text}")
    configure_file(cmake/LiveBuildProvenance.h.in "${live_generated}/LiveBuildProvenance.h" @ONLY)
    target_include_directories(live_runtime PUBLIC include source/live/runtime source/live/support)
    target_link_libraries(live_runtime PUBLIC live_proto live_uri)
    target_compile_definitions(live_runtime PRIVATE OMNIUSDLIVE_EXPORTS)
    set_target_properties(live_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden
                          VISIBILITY_INLINES_HIDDEN ON)
    if(MSVC)
        target_compile_options(live_runtime PRIVATE /W4 /WX)
    else()
        target_compile_options(live_runtime PRIVATE -Wall -Wextra -Werror -pedantic)
    endif()
    if(BUILD_TESTING)
        add_library(live_runtime_contract_probe SHARED)
        target_link_libraries(live_runtime_contract_probe PRIVATE live_runtime live_uri)
        if(UNIX AND NOT APPLE)
            target_link_options(live_runtime_contract_probe PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/Live.exports")
            set_property(TARGET live_runtime_contract_probe APPEND PROPERTY LINK_DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/Live.exports")
        endif()
        add_executable(live_runtime_foundation tests/native/runtime/runtime_foundation.cpp
                                              tests/native/runtime/request_lifecycle.cpp tests/native/runtime/auth_controller.cpp
                                              tests/native/runtime/layer_write_lifecycle.cpp tests/native/runtime/layer_read_lifecycle.cpp tests/native/runtime/layer_progress.cpp
                                              tests/native/runtime/jitter.cpp
                                              tests/native/runtime/drain.cpp
                                              tests/native/runtime/status.cpp
                                              tests/native/runtime/diagnostics.cpp
                                              tests/native/runtime/diagnostics_api.cpp
                                              tests/native/runtime/session_peers.cpp tests/native/runtime/session_api.cpp tests/native/runtime/layer_admission.cpp tests/native/runtime/layer_status.cpp)
        live_link_stack_dump(live_runtime_foundation)
        target_link_libraries(live_runtime_foundation PRIVATE live_runtime live_uri)
        set(live_diagnostic_catalog_header "${CMAKE_CURRENT_BINARY_DIR}/catalog-contract/LiveDiagnosticCatalogContract.h")
        add_custom_command(OUTPUT "${live_diagnostic_catalog_header}"
            COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/diagnostic_catalog.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/contracts/diagnostic-catalog.json"
                --header "${live_diagnostic_catalog_header}"
            DEPENDS tools/live/diagnostic_catalog.py tests/contracts/diagnostic-catalog.json
            VERBATIM)
        target_sources(live_runtime_foundation PRIVATE "${live_diagnostic_catalog_header}")
        target_include_directories(live_runtime_foundation PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/catalog-contract")
        # This executable includes the objects directly so it can test private
        # dependency leases; the shared probe separately checks the export ABI.
        target_compile_definitions(live_runtime_foundation PRIVATE OMNIUSDLIVE_EXPORTS)
        if(MSVC)
            target_compile_options(live_runtime_foundation PRIVATE /W4 /WX /UNDEBUG /wd4389) # C4389: see LiveWarnings.cmake
        else()
            target_compile_options(live_runtime_foundation PRIVATE -Wall -Wextra -Werror -pedantic -UNDEBUG)
        endif()
        add_test(NAME live_runtime_foundation COMMAND live_runtime_foundation)
        set_tests_properties(live_runtime_foundation PROPERTIES TIMEOUT 30)
        get_property(live_dependency_files GLOBAL PROPERTY LIVE_DEPENDENCY_FILES)
        set(live_dependency_args)
        foreach(dependency IN LISTS live_dependency_files)
            list(APPEND live_dependency_args --dependency "${dependency}")
        endforeach()
        set_property(TARGET live_runtime_foundation APPEND PROPERTY LINK_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveGrpcDependencies.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/Live.exports"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/write_build_manifest.py")
        add_custom_command(TARGET live_runtime_foundation POST_BUILD
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/write_build_manifest.py"
                --root "${CMAKE_CURRENT_SOURCE_DIR}" --artifact "$<TARGET_FILE:live_runtime_foundation>"
                --boundary runtime_foundation
                --input tests/contracts/diagnostic-catalog.json --input tools/live/diagnostic_catalog.py
                --compiler-id "${CMAKE_CXX_COMPILER_ID}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
                --configuration "$<CONFIG>" ${LIVE_SANITIZER_MANIFEST_ARGS}
                --input CMakeLists.txt --input include/omni/usd/live/Live.h
                --input source/live/runtime/LiveUri.h --input source/live/runtime/LiveUri.cpp
                --input source/live/runtime/LiveUriApi.cpp --input source/live/runtime/Live.exports
                --input source/live/runtime/LiveRuntime.h --input source/live/runtime/LiveRuntime.cpp --input source/live/support/LiveTsan.h
                --input source/live/runtime/LiveRuntimeApi.cpp --input cmake/LiveGrpcDependencies.cmake
                --input source/live/runtime/LiveRequest.h --input source/live/runtime/LiveRequest.cpp
                --input source/live/runtime/LiveUnary.h --input source/live/runtime/LiveUnary.cpp
                --input source/live/runtime/LiveDispatch.h --input source/live/runtime/LiveDispatch.cpp
                --input source/live/runtime/LiveMetricDuration.h --input source/live/runtime/LiveDeadline.h --input source/live/runtime/LiveExceptionResult.h
                --input source/live/runtime/LiveAuth.h --input source/live/runtime/LiveAuth.cpp
                --input source/live/runtime/LiveFinite.cpp --input source/live/runtime/LiveMetadata.h
                --input source/live/runtime/LiveLayerWrite.h --input source/live/runtime/LiveLayerWrite.cpp
                --input source/live/runtime/LiveLayerRead.h --input source/live/runtime/LiveLayerRead.cpp
                --input source/live/runtime/LiveLayerProgress.h --input source/live/runtime/LiveLayerProgress.cpp
                --input source/live/runtime/LiveLayerActor.h --input source/live/runtime/LiveLayerActor.cpp
                --input source/live/runtime/LiveJitter.h --input source/live/runtime/LiveJitter.cpp
                --input source/live/runtime/LiveDrain.h --input source/live/runtime/LiveDrain.cpp
                --input source/live/runtime/LiveStatus.h --input source/live/runtime/LiveStatus.cpp
                --input source/live/runtime/LiveDiagnostics.h --input source/live/runtime/LiveDiagnostics.cpp
                --input source/live/runtime/LiveDiagnosticsApi.cpp --input cmake/LiveBuildProvenance.h.in
                --input source/live/runtime/LiveSession.h --input source/live/runtime/LiveSession.cpp
                --input source/live/runtime/LiveSessionPeers.h --input source/live/runtime/LiveSessionPeers.cpp
                --input source/live/runtime/LiveSessionApi.cpp
                --input source/live/runtime/LiveLayerAdmission.h --input source/live/runtime/LiveLayerAdmission.cpp
                --input source/live/runtime/LiveLayerStatus.h --input source/live/runtime/LiveLayerStatus.cpp
                --input source/live/protos/nvidia/omniverse/live/v1alpha/live.proto
                --input tests/native/runtime/runtime_foundation.cpp
                --input tests/native/runtime/request_lifecycle.cpp
                --input tests/native/runtime/auth_controller.cpp
                --input tests/native/runtime/layer_write_lifecycle.cpp
                --input tests/native/runtime/layer_read_lifecycle.cpp
                --input tests/native/runtime/layer_progress.cpp --input tests/native/runtime/jitter.cpp --input tests/native/runtime/drain.cpp --input tests/native/runtime/status.cpp --input tests/native/runtime/diagnostics.cpp --input tests/native/runtime/diagnostics_api.cpp
                --input tests/native/runtime/session_peers.cpp --input tests/native/runtime/session_api.cpp --input tests/native/runtime/layer_admission.cpp --input tests/native/runtime/layer_status.cpp ${LIVE_TEST_SUPPORT_MANIFEST_ARGS} ${live_dependency_args}
            VERBATIM)
        add_executable(live_unary_driver tests/native/runtime/unary_driver.cpp tests/native/runtime/layer_read_driver.cpp tests/native/runtime/layer_actor_driver.cpp tests/native/runtime/diagnostics_driver.cpp tests/native/runtime/session_driver.cpp)
        live_link_stack_dump(live_unary_driver)
        target_link_libraries(live_unary_driver PRIVATE live_runtime live_uri)
        target_compile_definitions(live_unary_driver PRIVATE OMNIUSDLIVE_EXPORTS)
        if(MSVC)
            target_compile_options(live_unary_driver PRIVATE /W4 /WX /UNDEBUG /wd4389) # C4389: see LiveWarnings.cmake
        else()
            target_compile_options(live_unary_driver PRIVATE -Wall -Wextra -Werror -pedantic -UNDEBUG)
        endif()
        set_property(TARGET live_unary_driver APPEND PROPERTY LINK_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveGrpcDependencies.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/Live.exports"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/write_build_manifest.py")
        add_custom_command(TARGET live_unary_driver POST_BUILD
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/write_build_manifest.py"
                --root "${CMAKE_CURRENT_SOURCE_DIR}" --artifact "$<TARGET_FILE:live_unary_driver>"
                --boundary unary_transport
                --compiler-id "${CMAKE_CXX_COMPILER_ID}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
                --configuration "$<CONFIG>" ${LIVE_SANITIZER_MANIFEST_ARGS}
                --input CMakeLists.txt --input include/omni/usd/live/Live.h
                --input source/live/runtime/LiveUri.h --input source/live/runtime/LiveUri.cpp
                --input source/live/runtime/LiveUriApi.cpp --input source/live/runtime/Live.exports
                --input source/live/runtime/LiveRuntime.h --input source/live/runtime/LiveRuntime.cpp --input source/live/support/LiveTsan.h
                --input source/live/runtime/LiveRuntimeApi.cpp --input cmake/LiveGrpcDependencies.cmake
                --input source/live/runtime/LiveRequest.h --input source/live/runtime/LiveRequest.cpp
                --input source/live/runtime/LiveUnary.h --input source/live/runtime/LiveUnary.cpp
                --input source/live/runtime/LiveDispatch.h --input source/live/runtime/LiveDispatch.cpp
                --input source/live/runtime/LiveMetricDuration.h --input source/live/runtime/LiveDeadline.h --input source/live/runtime/LiveExceptionResult.h
                --input source/live/runtime/LiveAuth.h --input source/live/runtime/LiveAuth.cpp
                --input source/live/runtime/LiveFinite.cpp --input source/live/runtime/LiveMetadata.h
                --input source/live/runtime/LiveLayerWrite.h --input source/live/runtime/LiveLayerWrite.cpp
                --input source/live/runtime/LiveLayerRead.h --input source/live/runtime/LiveLayerRead.cpp
                --input source/live/runtime/LiveLayerProgress.h --input source/live/runtime/LiveLayerProgress.cpp
                --input source/live/runtime/LiveLayerActor.h --input source/live/runtime/LiveLayerActor.cpp
                --input source/live/runtime/LiveJitter.h --input source/live/runtime/LiveJitter.cpp
                --input source/live/runtime/LiveDrain.h --input source/live/runtime/LiveDrain.cpp
                --input source/live/runtime/LiveStatus.h --input source/live/runtime/LiveStatus.cpp
                --input source/live/runtime/LiveDiagnostics.h --input source/live/runtime/LiveDiagnostics.cpp
                --input source/live/runtime/LiveDiagnosticsApi.cpp --input cmake/LiveBuildProvenance.h.in
                --input source/live/runtime/LiveSession.h --input source/live/runtime/LiveSession.cpp
                --input source/live/runtime/LiveSessionPeers.h --input source/live/runtime/LiveSessionPeers.cpp
                --input source/live/runtime/LiveSessionApi.cpp
                --input source/live/runtime/LiveLayerAdmission.h --input source/live/runtime/LiveLayerAdmission.cpp
                --input source/live/runtime/LiveLayerStatus.h --input source/live/runtime/LiveLayerStatus.cpp
                --input source/live/protos/nvidia/omniverse/live/v1alpha/live.proto
                --input tests/native/runtime/unary_driver.cpp --input tests/native/runtime/layer_read_driver.cpp --input tests/native/runtime/layer_actor_driver.cpp --input tests/native/runtime/diagnostics_driver.cpp --input tests/native/runtime/session_driver.cpp ${LIVE_TEST_SUPPORT_MANIFEST_ARGS} ${live_dependency_args}
            VERBATIM)
        add_executable(live_grpc_schema_probe tests/native/runtime/grpc_schema_probe.cpp)
        live_link_stack_dump(live_grpc_schema_probe)
        target_link_libraries(live_grpc_schema_probe PRIVATE live_proto)
        if(MSVC)
            target_compile_options(live_grpc_schema_probe PRIVATE /UNDEBUG)
        else()
            target_compile_options(live_grpc_schema_probe PRIVATE -UNDEBUG)
        endif()
        add_test(NAME live_grpc_schema COMMAND live_grpc_schema_probe)
    endif()
    foreach(target live_runtime live_runtime_contract_probe live_runtime_foundation live_unary_driver live_proto live_grpc_schema_probe)
        live_sanitize(${target} PRIVATE)
    endforeach()
endif()

option(LIVE_BUILD_SCENE_TRANSPORT "Build actual scene journal/native actor integration" OFF)
if(LIVE_BUILD_SCENE_TRANSPORT)
    include(cmake/LiveSceneTransport.cmake)
endif()

option(LIVE_BUILD_PLUGIN "Build the shared Live Services USD plugin and public host boundary" OFF)
if(LIVE_BUILD_PLUGIN)
    include(cmake/LivePlugin.cmake)
endif()

option(LIVE_BUILD_PYTHON "Build the Python adapter linked to the same native plugin" OFF)
if(LIVE_BUILD_PYTHON)
    include(cmake/LivePython.cmake)
endif()

option(LIVE_BUILD_RESOLVER_RUNTIME "Build the live Ar resolver with the native runtime" OFF)
if(LIVE_BUILD_RESOLVER_RUNTIME)
    include(cmake/LiveResolverRuntime.cmake)
endif()
