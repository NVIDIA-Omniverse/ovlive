# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
if(NOT CMAKE_SYSTEM_NAME MATCHES "^(Linux|Windows|Darwin)$")
    message(FATAL_ERROR "LIVE_BUILD_PLUGIN supports the explicit Linux, Windows and macOS USD SDKs only")
endif()
foreach(required live_scene_transport live_runtime live_uri live_scene_dependencies)
    if(NOT TARGET ${required})
        message(FATAL_ERROR "LIVE_BUILD_PLUGIN requires LIVE_BUILD_SCENE_TRANSPORT and its explicit runtime/scene dependencies")
    endif()
endforeach()

# Registration translation units, and the creation bridge that shares their
# private layer header, are direct inputs: archive extraction cannot discard
# their static USD factories, and the bridge keeps the plugin's -Werror gate.
# Runtime/URI object libraries are likewise
# direct inputs so every public C entry point is present. The transport archive
# contains copies of those objects, but ordinary archive extraction leaves them
# unselected: there is exactly one runtime in this DSO. Do not whole-archive the
# scene dependencies (Progress and SHA256 also occur in component archives).
add_library(omni_usd_live SHARED
    source/live/scene/LiveUsd.cpp
    source/live/resolver/LiveResolver.cpp
    source/live/scene/LiveSdfFileFormat.cpp
    source/live/scene/UsdLayerCreation.cpp)
target_link_libraries(omni_usd_live PRIVATE live_runtime live_uri live_scene_transport)
live_link_usd(omni_usd_live PRIVATE ar plug)
target_include_directories(omni_usd_live
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>" "$<INSTALL_INTERFACE:include>")
target_compile_features(omni_usd_live PUBLIC cxx_std_17)
target_compile_definitions(omni_usd_live PRIVATE OMNIUSDLIVE_EXPORTS)
live_warnings(omni_usd_live USD)
set(live_plugin_exports "${CMAKE_CURRENT_SOURCE_DIR}/source/live/scene/plugin/LivePlugin.exports")
if(APPLE)
    # Mach-O exports the same public allowlist, with its leading symbol underscore.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${live_plugin_exports}")
    file(STRINGS "${live_plugin_exports}" live_export_lines REGEX "^[ ]+[^ #].*;$")
    set(live_macho_exports "${CMAKE_CURRENT_BINARY_DIR}/LivePlugin.exports.macos")
    file(WRITE "${live_macho_exports}" "")
    foreach(symbol IN LISTS live_export_lines)
        string(STRIP "${symbol}" symbol)
        string(REGEX REPLACE ";$" "" symbol "${symbol}")
        if(NOT symbol MATCHES "^local:")
            file(APPEND "${live_macho_exports}" "_${symbol}\n")
        endif()
    endforeach()
    target_link_options(omni_usd_live PRIVATE "LINKER:-exported_symbols_list,${live_macho_exports}" "LINKER:-undefined,error")
    set_property(TARGET omni_usd_live APPEND PROPERTY LINK_DEPENDS "${live_macho_exports}")
elseif(NOT WIN32)
    # The ELF plugin exports only the symbols the version script lists and links with no undefined symbol; a Windows
    # DLL exports the OMNI_USD_LIVE_API functions it defines.
    target_link_options(omni_usd_live PRIVATE "LINKER:--version-script=${live_plugin_exports}" "LINKER:-z,defs")
endif()
set_property(TARGET omni_usd_live APPEND PROPERTY LINK_DEPENDS "${live_plugin_exports}")
if(APPLE)
    set(live_loader_origin "@loader_path")
else()
    set(live_loader_origin "$ORIGIN")
endif()
set(live_plugin_package "${CMAKE_CURRENT_BINARY_DIR}/plugin/omni_usd_live")
set_target_properties(omni_usd_live PROPERTIES
    OUTPUT_NAME omni_usd_live CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${live_plugin_package}/lib/$<0:>"
    # A Windows DLL is a runtime artifact beside the executables that import it; its import library is an archive.
    RUNTIME_OUTPUT_DIRECTORY "${live_plugin_package}/bin/$<0:>"
    ARCHIVE_OUTPUT_DIRECTORY "${live_plugin_package}/lib/$<0:>"
    # Installed dependencies are supplied by the pinned host SDK, not copied or
    # silently resolved from developer build directories. Package-relative USD
    # metadata works after relocation; ABI/platform qualification is separate.
    INSTALL_RPATH "${live_loader_origin}" INSTALL_RPATH_USE_LINK_PATH FALSE)
# The package-relative plugin library that plugInfo.json names, as tools/live/build.py library() lays it out.
if(WIN32)
    set(LIVE_PLUGIN_LIBRARY "bin/omni_usd_live.dll")
else()
    set(LIVE_PLUGIN_LIBRARY "lib/libomni_usd_live${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/source/live/scene/plugin/plugInfo.json.in"
    "${live_plugin_package}/resources/plugInfo.json" @ONLY NEWLINE_STYLE UNIX)
set_property(TARGET omni_usd_live APPEND PROPERTY LINK_DEPENDS
    "${live_plugin_package}/resources/plugInfo.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LivePlugin.cmake")
add_custom_command(TARGET omni_usd_live POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${live_plugin_package}/include/omni/usd/live"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/include/omni/usd/live/Live.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/omni/usd/live/LiveUsd.h"
        "${live_plugin_package}/include/omni/usd/live"
    VERBATIM)
include(GNUInstallDirs)
install(TARGETS omni_usd_live LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/omni_usd_live/lib"
    RUNTIME DESTINATION "${CMAKE_INSTALL_LIBDIR}/omni_usd_live/bin"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/omni_usd_live/lib")
install(FILES "${live_plugin_package}/resources/plugInfo.json"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/omni_usd_live/resources")
install(FILES include/omni/usd/live/Live.h include/omni/usd/live/LiveUsd.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/omni/usd/live")
# The root license accompanies the client attribution and dependency records.
get_filename_component(live_repository_dir "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
install(FILES "${live_repository_dir}/LICENSE" "${CMAKE_CURRENT_SOURCE_DIR}/NOTICE"
    "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md" "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_LICENSES.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/BOM.json"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/omni_usd_live")
install(FILES source/live/protos/PROVENANCE.md
    DESTINATION "${CMAKE_INSTALL_DATADIR}/omni_usd_live/provenance" RENAME live-proto.md)
install(FILES source/live/wire/schema/provenance.json
    DESTINATION "${CMAKE_INSTALL_DATADIR}/omni_usd_live/provenance/wire-schema")
# The schema's Apache-2.0 notice retains its installed provenance path.
install(FILES source/live/wire/schema/LICENSE.txt
    DESTINATION "${CMAKE_INSTALL_DATADIR}/omni_usd_live/provenance/wire-schema" RENAME UPSTREAM-NOTICE.txt)
# The plugin contains FlatBuffers code; ship its Apache-2.0 license and pin.
install(FILES source/live/wire/third_party/flatbuffers/LICENSE.txt
    source/live/wire/third_party/flatbuffers/provenance.json
    DESTINATION "${CMAKE_INSTALL_DATADIR}/omni_usd_live/provenance/flatbuffers")
live_sanitize(omni_usd_live PRIVATE)
live_portable_python(omni_usd_live)

# Public C ABI consumer: the shared plugin owns the only runtime instance.
add_executable(live_diagnostics_sample
    source/live/samples/diagnostics/main.cpp
    source/live/samples/diagnostics/DiagnosticLog.cpp)
target_link_libraries(live_diagnostics_sample PRIVATE omni_usd_live Threads::Threads)
target_include_directories(live_diagnostics_sample PRIVATE source/live/samples/diagnostics)
target_compile_features(live_diagnostics_sample PRIVATE cxx_std_17)
live_warnings(live_diagnostics_sample)
set_target_properties(live_diagnostics_sample PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${live_plugin_package}/bin/$<0:>"
    BUILD_RPATH "${live_loader_origin}/../lib" INSTALL_RPATH "${live_loader_origin}/../lib;${live_loader_origin}/../lib/omni_usd_live/lib"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
install(TARGETS live_diagnostics_sample RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
live_sanitize(live_diagnostics_sample PRIVATE)

add_executable(live_participant_sample
    source/live/samples/participant/main.cpp
    source/live/samples/diagnostics/DiagnosticLog.cpp)
target_link_libraries(live_participant_sample PRIVATE omni_usd_live live_scene_dependencies)
live_link_usd(live_participant_sample PRIVATE js plug)
target_include_directories(live_participant_sample PRIVATE source/live/samples/diagnostics)
target_compile_features(live_participant_sample PRIVATE cxx_std_17)
live_warnings(live_participant_sample USD)
set_target_properties(live_participant_sample PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${live_plugin_package}/bin/$<0:>"
    BUILD_RPATH "${live_loader_origin}/../lib" INSTALL_RPATH "${live_loader_origin}/../lib;${live_loader_origin}/../lib/omni_usd_live/lib"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
install(TARGETS live_participant_sample RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
live_sanitize(live_participant_sample PRIVATE)
live_portable_python(live_participant_sample)

if(BUILD_TESTING)
    add_executable(live_participant_tests tests/native/participant/driver.cpp
        source/live/samples/diagnostics/DiagnosticLog.cpp)
    live_link_stack_dump(live_participant_tests)
    target_link_libraries(live_participant_tests PRIVATE omni_usd_live live_scene_dependencies)
    live_link_usd(live_participant_tests PRIVATE js plug)
    target_include_directories(live_participant_tests PRIVATE source/live/samples/diagnostics)
    target_compile_features(live_participant_tests PRIVATE cxx_std_17)
    live_warnings(live_participant_tests USD TEST)
    add_test(NAME live_participant_tests COMMAND live_participant_tests)
    set_tests_properties(live_participant_tests PROPERTIES TIMEOUT 30)
    live_sanitize(live_participant_tests PRIVATE)
    add_executable(live_diagnostic_log_tests
        tests/native/diagnostics_log/driver.cpp
        source/live/samples/diagnostics/DiagnosticLog.cpp)
    live_link_stack_dump(live_diagnostic_log_tests)
    target_include_directories(live_diagnostic_log_tests PRIVATE include source/live/samples/diagnostics)
    target_compile_features(live_diagnostic_log_tests PRIVATE cxx_std_17)
    live_warnings(live_diagnostic_log_tests TEST)
    add_test(NAME live_diagnostic_log_tests COMMAND live_diagnostic_log_tests)
    set_tests_properties(live_diagnostic_log_tests PROPERTIES TIMEOUT 30)
    add_executable(live_plugin_load_probe tests/native/plugin/load.cpp)
    live_link_stack_dump(live_plugin_load_probe)
    target_include_directories(live_plugin_load_probe PRIVATE include)
    target_link_libraries(live_plugin_load_probe PRIVATE live_scene_dependencies)
    live_link_usd(live_plugin_load_probe PRIVATE ar plug)
    target_compile_features(live_plugin_load_probe PRIVATE cxx_std_17)
    live_warnings(live_plugin_load_probe USD TEST)
    add_executable(live_plugin_driver tests/native/plugin/driver.cpp)
    live_link_stack_dump(live_plugin_driver)
    # This consumer gets the public headers and SDK only. In particular, it
    # must never link live_runtime/live_uri or any private component archive.
    target_link_libraries(live_plugin_driver PRIVATE omni_usd_live live_scene_dependencies)
    live_link_usd(live_plugin_driver PRIVATE ar plug usd)
    target_compile_features(live_plugin_driver PRIVATE cxx_std_17)
    live_warnings(live_plugin_driver USD TEST)
    target_include_directories(live_plugin_driver BEFORE PRIVATE "${live_plugin_package}/include")
    add_dependencies(live_plugin_driver live_plugin_load_probe live_diagnostics_sample live_diagnostic_log_tests live_participant_sample live_participant_tests)
    live_sanitize(live_diagnostic_log_tests PRIVATE)
    live_sanitize(live_plugin_load_probe PRIVATE)
    live_sanitize(live_plugin_driver PRIVATE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME live_diagnostic_log_json COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/diagnostics_log/check.py"
        "$<TARGET_FILE:live_diagnostic_log_tests>")
    set_tests_properties(live_diagnostic_log_json PROPERTIES TIMEOUT 30)
    add_test(NAME live_plugin_package COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/plugin/package.py"
        "${live_plugin_package}" "$<TARGET_FILE:live_plugin_load_probe>" "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(live_plugin_package PROPERTIES TIMEOUT 60)
    get_property(plugin_dependencies GLOBAL PROPERTY LIVE_DEPENDENCY_FILES)
    set(plugin_dependency_args)
    foreach(dependency IN LISTS plugin_dependencies)
        list(APPEND plugin_dependency_args --dependency "${dependency}")
    endforeach()
    file(GLOB_RECURSE plugin_identity_inputs CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/scene/*" "${CMAKE_CURRENT_SOURCE_DIR}/source/live/wire/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/runtime/*" "${CMAKE_CURRENT_SOURCE_DIR}/source/live/resolver/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/samples/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/protos/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/live/support/*" "${CMAKE_CURRENT_SOURCE_DIR}/include/omni/usd/live/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/*" "${CMAKE_CURRENT_SOURCE_DIR}/tests/reference_service/*.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/harness/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*")
    list(FILTER plugin_identity_inputs EXCLUDE REGEX "/__pycache__/|\\.py[co]$")
    file(GLOB_RECURSE plugin_sdk_inputs CONFIGURE_DEPENDS
        "${LIVE_FLATBUFFERS_ROOT}/include/*"
        "${LIVE_USD_ROOT}/include/*" "${LIVE_USD_ROOT}/lib/*"
        "${LIVE_USD_PYTHON_ROOT}/include/*" "${LIVE_USD_PYTHON_ROOT}/lib/*")
    # Provenance hashes whole SDK trees, including plugin metadata and headers
    # absent from compiler depfiles. A normal rebuild must refresh the artifact
    # and manifest after any recorded SDK change, not leave a stale dead end.
    set_property(TARGET omni_usd_live live_plugin_load_probe APPEND PROPERTY LINK_DEPENDS ${plugin_sdk_inputs})
    set_property(TARGET live_plugin_driver APPEND PROPERTY LINK_DEPENDS
        ${plugin_identity_inputs} ${plugin_dependencies} ${plugin_sdk_inputs}
        "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/docs/ovlive-design.md"
        "$<TARGET_FILE:omni_usd_live>" "$<TARGET_FILE:live_plugin_load_probe>"
        "$<TARGET_FILE:live_diagnostics_sample>" "$<TARGET_FILE:live_diagnostic_log_tests>"
        "$<TARGET_FILE:live_participant_sample>"
        "$<TARGET_FILE:live_participant_tests>"
        "${live_plugin_package}/resources/plugInfo.json")
    add_custom_command(TARGET live_plugin_driver POST_BUILD
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/plugin_identity.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}" --artifact "$<TARGET_FILE:live_plugin_driver>"
            --plugin "$<TARGET_FILE:omni_usd_live>" --plug-info "${live_plugin_package}/resources/plugInfo.json"
            --diagnostics-sample "$<TARGET_FILE:live_diagnostics_sample>"
            --participant-sample "$<TARGET_FILE:live_participant_sample>"
            --participant-tests "$<TARGET_FILE:live_participant_tests>"
            --diagnostic-log-tests "$<TARGET_FILE:live_diagnostic_log_tests>"
            --load-probe "$<TARGET_FILE:live_plugin_load_probe>" --flatbuffers "${LIVE_FLATBUFFERS_ROOT}"
            --usd "${LIVE_USD_ROOT}" --python "${LIVE_USD_PYTHON_ROOT}"
            --compiler-id "${CMAKE_CXX_COMPILER_ID}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
            --configuration "$<CONFIG>" ${LIVE_SANITIZER_MANIFEST_ARGS}
            ${plugin_dependency_args}
        VERBATIM)
endif()
