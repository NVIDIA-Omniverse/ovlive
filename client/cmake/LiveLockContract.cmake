# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This header-only lock contract needs no USD, Carbonite or transport SDK.
if(BUILD_TESTING)
    find_package(Threads REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_executable(live_lock_noalloc_test tests/native/core/lock_noalloc.cpp)
    live_link_stack_dump(live_lock_noalloc_test)
    target_include_directories(live_lock_noalloc_test PRIVATE source/live/support)
    target_compile_features(live_lock_noalloc_test PRIVATE cxx_std_17)
    target_link_libraries(live_lock_noalloc_test PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(live_lock_noalloc_test PRIVATE /W4 /WX /UNDEBUG /wd4389) # C4389: see LiveWarnings.cmake
    else()
        target_compile_options(live_lock_noalloc_test PRIVATE -Wall -Wextra -Werror -pedantic -UNDEBUG)
    endif()
    live_sanitize(live_lock_noalloc_test PRIVATE)
    set(lock_contract_inputs CMakeLists.txt cmake/LiveLockContract.cmake
        source/live/support/ReaderWriterLock.h
        source/live/support/ReaderWriterLock.inl
        tests/native/core/lock_noalloc.cpp tools/live/write_build_manifest.py)
    set(lock_contract_args "")
    foreach(input IN LISTS lock_contract_inputs)
        list(APPEND lock_contract_args --input "${input}")
        set_property(TARGET live_lock_noalloc_test APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${input}")
    endforeach()
    add_custom_command(TARGET live_lock_noalloc_test POST_BUILD
        COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_SOURCE_DIR}/tools/live/write_build_manifest.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}" --artifact "$<TARGET_FILE:live_lock_noalloc_test>"
            --boundary lock_noalloc --compiler-id "${CMAKE_CXX_COMPILER_ID}"
            --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
            --configuration "$<CONFIG>" ${LIVE_SANITIZER_MANIFEST_ARGS}
            ${lock_contract_args} ${LIVE_TEST_SUPPORT_MANIFEST_ARGS}
        VERBATIM)
    add_test(NAME live_lock_noalloc COMMAND live_lock_noalloc_test)
    set_tests_properties(live_lock_noalloc PROPERTIES TIMEOUT 10)
endif()
