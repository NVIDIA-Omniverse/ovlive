# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# __FILE__ strings (assertion messages included) and any debug information name the checkout, the build tree and the
# SDK roots by relative or symbolic prefixes, so no binary carries a path of the machine that built it. The options
# apply to every target of this directory tree. GCC applies the last matching map, so the longest prefix comes last;
# Clang prefers the longest; MSVC applies the first matching /pathmap, which it accepts only with
# /experimental:deterministic, so there the longest comes first.
set(live_path_maps "")
foreach(pair IN ITEMS "CMAKE_CURRENT_SOURCE_DIR=." "CMAKE_CURRENT_BINARY_DIR=<build>" "LIVE_USD_ROOT=<usd>"
                      "LIVE_USD_PYTHON_ROOT=<python>" "LIVE_DEPENDENCY_ROOT=<deps>" "LIVE_PYBIND11_ROOT=<pybind11>")
    string(REPLACE "=" ";" pair "${pair}")
    list(GET pair 0 variable)
    list(GET pair 1 replacement)
    if(NOT "${${variable}}" STREQUAL "")
        file(REAL_PATH "${${variable}}" resolved)
        foreach(path IN ITEMS "${${variable}}" "${resolved}")
            string(REGEX REPLACE "/+$" "" path "${path}")
            string(LENGTH "${path}" length)
            # Zero-padded length first, so sorting orders the maps from the shortest prefix to the longest.
            string(PREPEND length "0000")
            string(REGEX MATCH "....$" length "${length}")
            list(APPEND live_path_maps "${length}|${path}=${replacement}")
        endforeach()
    endif()
endforeach()
# Each dependency package is usually a link into a package cache, and GCC names a system header by its resolved
# path when that path is shorter, so each package's resolved directory maps to its name under <deps>.
if(NOT "${LIVE_DEPENDENCY_ROOT}" STREQUAL "")
    file(GLOB live_dependency_packages LIST_DIRECTORIES true "${LIVE_DEPENDENCY_ROOT}/*")
    foreach(package IN LISTS live_dependency_packages)
        if(IS_DIRECTORY "${package}")
            file(REAL_PATH "${package}" resolved)
            get_filename_component(name "${package}" NAME)
            string(LENGTH "${resolved}" length)
            string(PREPEND length "0000")
            string(REGEX MATCH "....$" length "${length}")
            list(APPEND live_path_maps "${length}|${resolved}=<deps>/${name}")
        endif()
    endforeach()
endif()
list(REMOVE_DUPLICATES live_path_maps)
list(SORT live_path_maps)
if(MSVC)
    list(REVERSE live_path_maps)
    add_compile_options(/experimental:deterministic)
endif()
foreach(map IN LISTS live_path_maps)
    string(REGEX REPLACE "^[0-9]+\\|" "" map "${map}")
    if(MSVC)
        add_compile_options("/pathmap:${map}")
    else()
        add_compile_options("-ffile-prefix-map=${map}")
    endif()
endforeach()
# GCC also rewrites a system header's path to its canonical form when that is shorter, which for a dependency
# package linked into the dependency root is the package manager's cache path; with the rewriting off, a header
# keeps the spelling of the include directory it was found in, for the SDK roots as well as the packages.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options("-fno-canonical-system-headers")
endif()
# A build tree RPATH names the build tree, and installation rewrites only the RPATH string: the build ID and the
# dynamic string table size still depend on the build directory. A packaging build links with the install RPATH
# (CMAKE_BUILD_WITH_INSTALL_RPATH), so its binaries are independent of the build directory, and the linker finds the
# SDK libraries that the plugin needs through -rpath-link, which the binaries do not record.
if(CMAKE_BUILD_WITH_INSTALL_RPATH AND UNIX AND NOT APPLE)
    foreach(root IN ITEMS "${LIVE_USD_ROOT}" "${LIVE_USD_PYTHON_ROOT}")
        if(NOT root STREQUAL "")
            add_link_options("LINKER:-rpath-link,${root}/lib")
        endif()
    endforeach()
endif()
