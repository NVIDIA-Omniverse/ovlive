// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <omni/usd/live/Live.h>
#include <pxr/base/plug/registry.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/ar/resolver.h>
#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    include <tlhelp32.h>
#    include <filesystem>
#else
#    include <dlfcn.h>
#    if defined(__APPLE__)
#        include <mach-o/dyld.h>
#    endif
#endif
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
#if defined(_WIN32)
using Library = HMODULE;
Library openLibrary(const char* path) { return LoadLibraryW(std::filesystem::path(path).wstring().c_str()); }
// The module the process already holds for `path`, without loading it.
Library loadedLibrary(const char* path) { return GetModuleHandleW(std::filesystem::path(path).wstring().c_str()); }
const char* openError() { return "LoadLibraryW failed"; }
void* lookup(Library library, const char* name) { return reinterpret_cast<void*>(GetProcAddress(library, name)); }
// The file names of every module the process has loaded.
std::vector<std::string> loadedModules() {
    std::vector<std::string> names;
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    check(snapshot != INVALID_HANDLE_VALUE, "could not list loaded modules");
    MODULEENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    for (bool more = Module32FirstW(snapshot, &entry); more; more = Module32NextW(snapshot, &entry))
        names.push_back(std::filesystem::path(entry.szExePath).filename().string());
    CloseHandle(snapshot);
    return names;
}
#else
using Library = void*;
Library openLibrary(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
Library loadedLibrary(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD); }
const char* openError() { return dlerror(); }
void* lookup(Library library, const char* name) { return dlsym(library, name); }
std::vector<std::string> loadedModules() {
    std::vector<std::string> lines;
#    if defined(__APPLE__)
    for (uint32_t index = 0; index < _dyld_image_count(); ++index) {
        const char* name = _dyld_get_image_name(index);
        check(name, "could not identify loaded image");
        lines.emplace_back(name);
    }
#    else
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) lines.push_back(line);
#    endif
    return lines;
}
#endif
template<class Function> Function symbol(Library library, const char* name) {
    const auto value = lookup(library, name);
    check(value, name);
    return reinterpret_cast<Function>(value);
}
}
int main(int argc, char** argv) {
    try {
        check(argc == 4, "usage: load-probe library metadata api-first|plugin-first");
        Library library = nullptr;
        const bool apiFirst = std::string(argv[3]) == "api-first";
        check(apiFirst || std::string(argv[3]) == "plugin-first", "invalid load order");
        if (apiFirst) {
            library = openLibrary(argv[1]);
            check(library, openError());
        }
        PlugRegistry::GetInstance().RegisterPlugins(argv[2]);
        const auto format = SdfFileFormat::FindById(TfToken("live"));
        check(bool(format) && format->GetFormatId() == TfToken("live"), "packaged format did not register");
        if (!apiFirst) {
            // FindById must have loaded exactly this canonical library already.
            library = loadedLibrary(argv[1]);
            check(library, "USD did not load recorded library");
        }
        auto initialize = symbol<decltype(&omniUsdLiveInitialize)>(library, "omniUsdLiveInitialize");
        auto shutdown = symbol<decltype(&omniUsdLiveShutdown)>(library, "omniUsdLiveShutdown");
        auto registerEndpoint = symbol<decltype(&omniUsdLiveRegisterEndpoint)>(library, "omniUsdLiveRegisterEndpoint");
        auto unregisterEndpoint = symbol<decltype(&omniUsdLiveUnregisterEndpoint)>(library,
            "omniUsdLiveUnregisterEndpoint");
        check(initialize(nullptr) == OMNI_USD_LIVE_OK, "initialize failed");
        OmniUsdLiveEndpointOptions options{};
        options.struct_size = sizeof(options);
        options.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        options.authority = "127.0.0.1:35497";
        OmniUsdLiveEndpoint endpoint = 0;
        check(registerEndpoint(&options, &endpoint) == OMNI_USD_LIVE_OK, "endpoint registration failed");
        const std::string uri = "live://127.0.0.1:35497/session/root.live";
        check(ArGetResolver().ResolveForNewAsset(uri).GetPathString() == uri,
            "USD and public API have different runtimes");
        check(unregisterEndpoint(endpoint, 5000) == OMNI_USD_LIVE_OK, "endpoint cleanup failed");
        check(shutdown(5000) == OMNI_USD_LIVE_OK, "runtime cleanup failed");
        for (const auto& module : loadedModules()) {
            check(module.find("omniclient") == std::string::npos, "omniclient loaded");
            check(module.find("live_resolver_runtime_component") == std::string::npos,
                "second resolver runtime loaded");
        }
        std::cout << "shared-plugin load-order and one-runtime probe passed\n";
        // USD owns registered factories for process lifetime; never unload the library.
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
