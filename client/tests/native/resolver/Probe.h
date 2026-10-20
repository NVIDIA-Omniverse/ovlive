// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>
#include <vector>

namespace live_resolver_probe {
struct Event { std::string hook, path, detail; };
// Test-only process-local observations, shared by the plugin and its driver.
void clear();
std::vector<Event> events();
void registered(bool value);
void creationMode(std::string);
std::string capturedComment();
bool capturedTargetExpired();
void marker(const char*);
void enqueueAnonymous();
size_t drainDeferred();
}
