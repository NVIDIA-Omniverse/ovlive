# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
if(NOT TARGET omni_usd_live OR NOT CMAKE_SYSTEM_NAME MATCHES "^(Linux|Windows|Darwin)$")
    message(FATAL_ERROR "LIVE_BUILD_PYTHON requires the explicit Linux, Windows or macOS shared plugin")
endif()
if(LIVE_SANITIZED)
    message(FATAL_ERROR "Python needs a separately provisioned sanitizer-compatible interpreter; no sanitizer Python lane is qualified")
endif()
set(LIVE_PYBIND11_ROOT "" CACHE PATH "Explicit pybind11 2.11.1 headers")
set(LIVE_PYTHON_EXECUTABLE "${LIVE_PYTHON_INTERPRETER}" CACHE FILEPATH "Python interpreter matching the selected SDK headers")
if(NOT EXISTS "${LIVE_PYBIND11_ROOT}/include/pybind11/pybind11.h")
    message(FATAL_ERROR "LIVE_PYBIND11_ROOT must supply pinned 2.11.1 headers")
endif()
foreach(version MAJOR MINOR PATCH)
    file(STRINGS "${LIVE_PYBIND11_ROOT}/include/pybind11/detail/common.h" pybind_${version}
        REGEX "^#define PYBIND11_VERSION_${version} ")
endforeach()
if(NOT pybind_MAJOR MATCHES " 2$" OR NOT pybind_MINOR MATCHES " 11$" OR NOT pybind_PATCH MATCHES " 1$")
    message(FATAL_ERROR "Expected pybind11 2.11.1")
endif()
execute_process(COMMAND "${LIVE_PYTHON_EXECUTABLE}" -c
    "import sys,sysconfig; assert sys.version_info[:2] == (3,12); print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE live_python_suffix OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT live_python_suffix MATCHES "^\\.(cpython-312-[A-Za-z0-9_-]+\\.so|cp312-win_amd64\\.pyd)$")
    message(FATAL_ERROR "Unsupported Python extension suffix: ${live_python_suffix}")
endif()
set(live_python_source "${CMAKE_CURRENT_SOURCE_DIR}/source/bindings/python/omni/usd/live")
set(live_python_package "${live_plugin_package}/python/omni/usd/live")
file(GLOB live_python_cpp CONFIGURE_DEPENDS "${live_python_source}/*.cpp")
add_library(omni_usd_live_python MODULE ${live_python_cpp})
target_link_libraries(omni_usd_live_python PRIVATE omni_usd_live Threads::Threads ${CMAKE_DL_LIBS})
live_link_usd(omni_usd_live_python PRIVATE sdf tf python)
if(APPLE)
    target_link_libraries(omni_usd_live_python PRIVATE "${LIVE_PYTHON_LIBRARY}")
elseif(WIN32)
    # Python's pyconfig.h requests its import library by name (#pragma comment(lib)).
    target_link_libraries(omni_usd_live_python PRIVATE "${LIVE_PYTHON_LIBRARY}")
    get_filename_component(live_python_libs "${LIVE_PYTHON_LIBRARY}" DIRECTORY)
    target_link_directories(omni_usd_live_python PRIVATE "${live_python_libs}")
endif()
target_include_directories(omni_usd_live_python PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(omni_usd_live_python SYSTEM PRIVATE
    "${LIVE_PYBIND11_ROOT}/include" "${LIVE_PYTHON_INCLUDE}" "${LIVE_USD_ROOT}/include")
target_compile_features(omni_usd_live_python PRIVATE cxx_std_17)
live_warnings(omni_usd_live_python USD)
set_target_properties(omni_usd_live_python PROPERTIES
    PREFIX "" OUTPUT_NAME "_live" SUFFIX "${live_python_suffix}"
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${live_python_package}/$<0:>"
    BUILD_RPATH "${live_loader_origin}/../../../../lib" INSTALL_RPATH "${live_loader_origin}/../../../../lib"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
live_portable_python(omni_usd_live_python)
file(GLOB live_python_modules CONFIGURE_DEPENDS "${live_python_source}/*.py" "${live_python_source}/*.pyi" "${live_python_source}/py.typed")
foreach(module IN LISTS live_python_modules)
    get_filename_component(name "${module}" NAME)
    configure_file("${module}" "${live_python_package}/${name}" COPYONLY)
endforeach()
install(TARGETS omni_usd_live_python LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/omni_usd_live/python/omni/usd/live")
install(FILES ${live_python_modules} DESTINATION "${CMAKE_INSTALL_LIBDIR}/omni_usd_live/python/omni/usd/live")
set(live_kit_source "${CMAKE_CURRENT_SOURCE_DIR}/source/kit")
configure_file("${live_kit_source}/config/extension.toml" "${live_plugin_package}/config/extension.toml" COPYONLY)
configure_file("${live_kit_source}/config/live-sdk.json.in" "${live_plugin_package}/config/live-sdk.json" @ONLY NEWLINE_STYLE UNIX)
file(GLOB_RECURSE live_kit_modules CONFIGURE_DEPENDS "${live_kit_source}/python/*.py")
foreach(module IN LISTS live_kit_modules)
    file(RELATIVE_PATH relative "${live_kit_source}" "${module}")
    configure_file("${module}" "${live_plugin_package}/${relative}" COPYONLY)
    get_filename_component(directory "${relative}" DIRECTORY)
    install(FILES "${module}" DESTINATION "${CMAKE_INSTALL_LIBDIR}/omni_usd_live/${directory}")
endforeach()
install(FILES "${live_plugin_package}/config/extension.toml" "${live_plugin_package}/config/live-sdk.json"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/omni_usd_live/config")
if(APPLE)
    set(live_library_path_variable DYLD_LIBRARY_PATH)
else()
    set(live_library_path_variable LD_LIBRARY_PATH)
endif()
if(BUILD_TESTING)
    add_dependencies(omni_usd_live_python live_plugin_driver)
    file(GLOB_RECURSE live_python_sdk_inputs CONFIGURE_DEPENDS "${LIVE_PYBIND11_ROOT}/include/*")
    set_property(TARGET omni_usd_live_python APPEND PROPERTY LINK_DEPENDS
        ${live_python_sdk_inputs} ${live_python_modules} ${live_kit_modules}
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LivePython.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/python_identity.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/python/package.py"
        "${live_plugin_package}/config/extension.toml" "${live_plugin_package}/config/live-sdk.json"
        "$<TARGET_FILE:live_plugin_driver>" "$<TARGET_FILE:omni_usd_live>")
    add_custom_command(TARGET omni_usd_live_python POST_BUILD
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/python_identity.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}" --artifact "$<TARGET_FILE:omni_usd_live_python>"
            --plugin-driver "$<TARGET_FILE:live_plugin_driver>"
            --pybind "${LIVE_PYBIND11_ROOT}" --interpreter "${LIVE_PYTHON_EXECUTABLE}"
        VERBATIM)
    add_test(NAME live_python_package COMMAND ${CMAKE_COMMAND} -E env
        "${live_library_path_variable}=${LIVE_USD_ROOT}/lib:${LIVE_USD_PYTHON_ROOT}/lib"
        "PYTHONDONTWRITEBYTECODE=1" "PYTHONNOUSERSITE=1"
        "${LIVE_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/python/package.py"
        "${live_plugin_package}" "${LIVE_USD_ROOT}")
    set_tests_properties(live_python_package PROPERTIES TIMEOUT 45)
    add_test(NAME live_installed_package COMMAND ${CMAKE_COMMAND} -E env
        "${live_library_path_variable}=${LIVE_USD_ROOT}/lib:${LIVE_USD_PYTHON_ROOT}/lib"
        "PYTHONDONTWRITEBYTECODE=1" "PYTHONNOUSERSITE=1"
        "${LIVE_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/python/installed_package.py"
        --cmake "${CMAKE_COMMAND}" --build-dir "${CMAKE_BINARY_DIR}"
        --usd-root "${LIVE_USD_ROOT}" --config "$<CONFIG>")
    set_tests_properties(live_installed_package PROPERTIES TIMEOUT 60)
endif()
