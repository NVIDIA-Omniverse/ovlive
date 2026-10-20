# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Shared native test support (tests/native/support), recorded in every native
# test driver's identity manifest.
file(GLOB LIVE_TEST_SUPPORT_INPUTS LIST_DIRECTORIES false CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/support/*")
list(APPEND LIVE_TEST_SUPPORT_INPUTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiveTestSupport.cmake")
list(SORT LIVE_TEST_SUPPORT_INPUTS)
set(LIVE_TEST_SUPPORT_MANIFEST_ARGS)
foreach(input IN LISTS LIVE_TEST_SUPPORT_INPUTS)
    file(RELATIVE_PATH relative "${CMAKE_CURRENT_SOURCE_DIR}" "${input}")
    list(APPEND LIVE_TEST_SUPPORT_MANIFEST_ARGS --input "${relative}")
endforeach()
# Opt-in thread-stack dump for native test drivers; never linked into a sample
# or the plugin library.
if(BUILD_TESTING)
    add_library(live_test_stack_dump OBJECT tests/native/support/StackDump.cpp)
    target_compile_features(live_test_stack_dump PRIVATE cxx_std_17)
    set_target_properties(live_test_stack_dump PROPERTIES POSITION_INDEPENDENT_CODE ON)
    live_sanitize(live_test_stack_dump PRIVATE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_package(Threads REQUIRED)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        add_executable(live_stack_dump_probe tests/native/support/stack_dump_probe.cpp)
        target_link_libraries(live_stack_dump_probe PRIVATE live_test_stack_dump Threads::Threads)
        set_target_properties(live_stack_dump_probe PROPERTIES ENABLE_EXPORTS ON)
        live_sanitize(live_stack_dump_probe PRIVATE)
        add_test(NAME live_stack_dump COMMAND ${Python3_EXECUTABLE} -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/support/stack_dump_check.py"
            $<TARGET_FILE:live_stack_dump_probe> ${LIVE_SANITIZER_KIND})
        set_tests_properties(live_stack_dump PROPERTIES TIMEOUT 30)
    endif()
endif()
# Links the stack dump into a test driver when tests are built; the driver
# relinks, and so refreshes its identity manifest, when the support inputs change.
function(live_link_stack_dump target)
    if(TARGET live_test_stack_dump)
        target_link_libraries(${target} PRIVATE live_test_stack_dump)
        set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS ${LIVE_TEST_SUPPORT_INPUTS})
    endif()
endfunction()
