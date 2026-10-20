// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>

namespace threading {

void tprintf(std::stringstream& stm, const char* format); // base function

template<typename T, typename... Targs>
void tprintf(std::stringstream& stm, const char* format, T value, Targs... Fargs) // recursive variadic function
{
    if (format != nullptr) {
        for (; *format != '\0'; format++) {
            if (*format == '%') {
                stm << value;
                tprintf(stm, format + 1, Fargs...); // recursive call
                return;
            }
            stm << *format;
        }
    }
}

template<typename T, typename... Targs>
std::string tprintf(const char* format, T value, Targs... Fargs) // recursive variadic function
{
    std::stringstream stm;
    tprintf(stm, format, value, Fargs...);
    return stm.str();
}
} // namespace threading
