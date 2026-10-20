# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# live_warnings(<target> [USD] [TEST]) applies the project's warning gate to one target.
# GCC and Clang treat every -Wall -Wextra warning as an error; USD targets also allow deprecated declarations, which
# the USD headers use. MSVC treats /W4 warnings as errors in targets that include no USD header; USD's own headers
# warn under MSVC, so USD targets build at /W3 without /WX, allowing deprecated declarations (C4996). TEST keeps
# assertions enabled in a Release build.
function(live_warnings target)
    cmake_parse_arguments(PARSE_ARGV 1 live "USD;TEST" "" "")
    if(MSVC)
        if(live_USD)
            target_compile_options(${target} PRIVATE /W3 /wd4996)
        else()
            target_compile_options(${target} PRIVATE /W4 /WX)
        endif()
        if(live_TEST)
            # MSVC types C enumerators as int, so tests comparing an unsigned result with one warn (C4389); GCC and
            # Clang do not warn on a constant that fits.
            target_compile_options(${target} PRIVATE /UNDEBUG /wd4389)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Werror)
        if(live_USD)
            target_compile_options(${target} PRIVATE -Wno-deprecated)
        endif()
        if(live_TEST)
            target_compile_options(${target} PRIVATE -UNDEBUG)
        endif()
    endif()
endfunction()
