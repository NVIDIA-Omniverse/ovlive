// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// A runtime the process still owns at exit remains owned after the library's
// static destruction (Python guide, teardown), so threads that outlive it, such
// as an interpreter's abandoned-owner cleanup, still reach a live runtime.
#include "omni/usd/live/Live.h"
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>

namespace {
using Shutdown = OmniUsdLiveResult (*)(uint32_t);
using Initialize = OmniUsdLiveResult (*)(const OmniUsdLiveRuntimeOptions*);
using RegisterEndpoint = OmniUsdLiveResult (*)(const OmniUsdLiveEndpointOptions*, OmniUsdLiveEndpoint*);

Shutdown shutdownAtExit = nullptr;

// Registered before the library loads, so it runs after every static
// destructor the library registered. It judges only a run that reached main's
// successful return; a failed setup keeps its own exit status.
void afterLibraryDestruction() {
    if (!shutdownAtExit) return;
    const auto result = shutdownAtExit(0);
    std::printf("shutdown after static destruction: %d\n", static_cast<int>(result));
    std::fflush(stdout);
    _exit(result == OMNI_USD_LIVE_BUSY ? 0 : 1);
}

template<class T> T symbol(void* library, const char* name) {
    return reinterpret_cast<T>(dlsym(library, name));
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: live_runtime_exit <runtime library>\n");
        return 2;
    }
    std::atexit(afterLibraryDestruction);
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        std::fprintf(stderr, "%s\n", dlerror());
        return 2;
    }
    const auto initialize = symbol<Initialize>(library, "omniUsdLiveInitialize");
    const auto registerEndpoint = symbol<RegisterEndpoint>(library, "omniUsdLiveRegisterEndpoint");
    const auto shutdown = symbol<Shutdown>(library, "omniUsdLiveShutdown");
    if (!initialize || !registerEndpoint || !shutdown) return 2;
    if (initialize(nullptr) != OMNI_USD_LIVE_OK) return 3;
    auto options = OmniUsdLiveEndpointOptions{};
    options.struct_size = sizeof(options);
    options.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    options.authority = "127.0.0.1:1";
    OmniUsdLiveEndpoint endpoint = 0;
    if (registerEndpoint(&options, &endpoint) != OMNI_USD_LIVE_OK) return 4;
    // The registered endpoint keeps the runtime owned through exit.
    if (shutdown(0) != OMNI_USD_LIVE_BUSY) return 5;
    shutdownAtExit = shutdown;
    return 0;
}
