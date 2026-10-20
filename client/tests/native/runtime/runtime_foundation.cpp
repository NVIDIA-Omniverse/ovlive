// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveRuntime.h"
#include "LiveDiagnostics.h"
#include "LiveStatus.h"
#include "LiveMetadata.h"
#include "LiveDeadline.h"
#include <grpcpp/alarm.h>

#include <cassert>
#include <cstddef>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <vector>
#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace omni::usd::live;

namespace {
void rpcMonotonicDeadline() {
    // Fixed gRPC-clock samples; no wall-clock dependency or passing-path wait.
    const gpr_timespec now{100, 900000000, GPR_CLOCK_MONOTONIC};
    struct Case { int64_t budgetMs, seconds; int32_t nanos; };
    for (const auto value : {Case{0,100,900000000}, Case{250,101,150000000},
                            Case{-950,99,950000000}, Case{UINT32_MAX,4295068,195000000}}) {
        grpc::ClientContext context;
        context.set_deadline(rpcDeadlineAfter(std::chrono::milliseconds(value.budgetMs), now));
        const auto deadline = context.raw_deadline();
        assert(deadline.clock_type == GPR_CLOCK_MONOTONIC);
        assert(deadline.tv_sec == value.seconds && deadline.tv_nsec == value.nanos);
        // Already exhausted finite-operation budgets remain expired. Context
        // preparation cannot silently restart a fresh default timeout.
        assert((gpr_time_cmp(deadline, now) <= 0) == (value.budgetMs <= 0));
    }
}
void metadataDeadlineInheritance() {
    using Time = MetadataDeadlineScope::TimePoint;
    const auto outer = Time{} + std::chrono::seconds(20);
    const auto inner = Time{} + std::chrono::seconds(10);
    assert(MetadataDeadlineScope::limit(Time::max()) == Time::max());
    {
        MetadataDeadlineScope scope(outer);
        assert(MetadataDeadlineScope::limit(Time::max()) == outer);
        { MetadataDeadlineScope larger(Time::max()); assert(MetadataDeadlineScope::limit(Time::max()) == outer); }
        try { MetadataDeadlineScope smaller(inner); assert(MetadataDeadlineScope::limit(outer) == inner); throw 1; }
        catch (int) {}
        assert(MetadataDeadlineScope::limit(Time::max()) == outer);
        auto independent = std::async(std::launch::async, [] { return MetadataDeadlineScope::limit(Time::max()); });
        assert(independent.get() == Time::max());
    }
    assert(MetadataDeadlineScope::limit(Time::max()) == Time::max());
    auto untouched = OmniUsdLiveLayerInfo{};
    untouched.created_time_ms = 123; untouched.modified_time_ms = 456;
    uint64_t modifiedUs = 789;
    auto endpoint = [](const char* authority, const char* token) {
        auto options = OmniUsdLiveEndpointOptions{};
        options.struct_size = sizeof(options);
        options.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        options.authority = authority;
        options.access_token = token;
        OmniUsdLiveEndpoint handle = 0;
        assert(omniUsdLiveRegisterEndpoint(&options, &handle) == OMNI_USD_LIVE_OK);
        return handle;
    };
    const auto usable = endpoint("127.0.0.1:1", "synthetic-writer");
    const auto starved = endpoint("127.0.0.1:2", "");
    {
        MetadataDeadlineScope expired(Time::min());
        // Expiry is explicit input, not elapsed time. An invalid URI or an
        // unregistered authority is rejected first; a resolved one is
        // classified by its endpoint's credential before any transport is
        // allocated. Caller output cannot change.
        assert(statLayerMetadata("not-a-uri", untouched, &modifiedUs) == OMNI_USD_LIVE_INVALID_ARGUMENT);
        assert(statLayerMetadata("live://127.0.0.1:3/s/root.live", untouched, &modifiedUs) == OMNI_USD_LIVE_NOT_FOUND);
        assert(statLayerMetadata("live://127.0.0.1:1/s/root.live", untouched, &modifiedUs) == OMNI_USD_LIVE_TIMEOUT);
        assert(statLayerMetadata("live://127.0.0.1:2/s/root.live", untouched, &modifiedUs) ==
            OMNI_USD_LIVE_UNAUTHENTICATED);
        assert(untouched.created_time_ms == 123 && untouched.modified_time_ms == 456 && modifiedUs == 789);
    }
    assert(omniUsdLiveUnregisterEndpoint(usable, 0) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterEndpoint(starved, 0) == OMNI_USD_LIVE_OK);
}
template <class T> T record()
{
    T value{};
    value.struct_size = sizeof(value);
    value.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    return value;
}

OmniUsdLiveEndpointOptions endpointOptions(const char* authority)
{
    auto value = record<OmniUsdLiveEndpointOptions>();
    value.authority = authority;
    return value;
}

OmniUsdLiveEndpoint add(const char* authority)
{
    auto options = endpointOptions(authority);
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&options, &endpoint) == OMNI_USD_LIVE_OK);
    assert(endpoint);
    return endpoint;
}

void finish()
{
    auto result = omniUsdLiveShutdown(0);
    assert(result == OMNI_USD_LIVE_OK || result == OMNI_USD_LIVE_TIMEOUT);
    if (result == OMNI_USD_LIVE_TIMEOUT)
        assert(omniUsdLiveShutdown(2000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_NOT_INITIALIZED);
}

void staticCapabilities()
{
    assert(omniUsdLiveGetCapabilities(nullptr)==OMNI_USD_LIVE_INVALID_ARGUMENT);
    auto capabilities=record<OmniUsdLiveCapabilities>();
    capabilities.struct_size--;
    auto before=capabilities;
    assert(omniUsdLiveGetCapabilities(&capabilities)==OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(std::memcmp(&capabilities,&before,sizeof(before))==0);
    capabilities=record<OmniUsdLiveCapabilities>(); capabilities.abi_version=2u<<16;
    before=capabilities;
    assert(omniUsdLiveGetCapabilities(&capabilities)==OMNI_USD_LIVE_UNSUPPORTED);
    assert(std::memcmp(&capabilities,&before,sizeof(before))==0);
    struct Extended { OmniUsdLiveCapabilities known; uint64_t trailing=UINT64_C(0x12785634abcdef90); };
    Extended extended;
    extended.known=record<OmniUsdLiveCapabilities>();
    extended.known.struct_size=sizeof(extended);
    extended.known.abi_version=OMNI_USD_LIVE_ABI_VERSION|0xffff; // Same-major extension.
    assert(omniUsdLiveGetCapabilities(&extended.known)==OMNI_USD_LIVE_OK);
    assert(extended.known.struct_size==sizeof(extended) && extended.known.abi_version==OMNI_USD_LIVE_ABI_VERSION);
    assert(extended.known.delegate_version==2 && extended.known.autonomous_progress==1);
    assert(extended.known.diagnostic_schema_version==OMNI_USD_LIVE_DIAGNOSTIC_SCHEMA_VERSION);
    assert(extended.trailing==UINT64_C(0x12785634abcdef90));
    // Static queries neither require nor acquire an initialization reference.
    assert(!getRuntime());
    assert(omniUsdLiveInitialize(nullptr)==OMNI_USD_LIVE_OK);
    capabilities=record<OmniUsdLiveCapabilities>();
    assert(omniUsdLiveGetCapabilities(&capabilities)==OMNI_USD_LIVE_OK);
    const auto initialized=capabilities;
    finish();
    capabilities=record<OmniUsdLiveCapabilities>();
    assert(omniUsdLiveGetCapabilities(&capabilities)==OMNI_USD_LIVE_OK);
    assert(std::memcmp(&capabilities,&initialized,sizeof(capabilities))==0 && !getRuntime());
}

void abiAndConfiguration()
{
    assert(omniUsdLiveGetAbiVersion(nullptr) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    uint32_t version = 0;
    assert(omniUsdLiveGetAbiVersion(&version) == OMNI_USD_LIVE_OK && version == OMNI_USD_LIVE_ABI_VERSION);
    const char* text = nullptr;
    assert(omniUsdLiveGetResultString(OMNI_USD_LIVE_OUTCOME_UNKNOWN, &text) == OMNI_USD_LIVE_OK);
    assert(std::strcmp(text, "OUTCOME_UNKNOWN") == 0);
    assert(omniUsdLiveGetResultString(UINT32_MAX, &text) == OMNI_USD_LIVE_OK);
    assert(std::strcmp(text, "UNKNOWN_RESULT") == 0);
    assert(omniUsdLiveGetResultString(17u, &text) == OMNI_USD_LIVE_OK && std::string(text) == "UNKNOWN_RESULT");
    assert(omniUsdLiveGetResultString(0, nullptr) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.struct_size--;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    options = record<OmniUsdLiveRuntimeOptions>();
    options.abi_version = 2U << 16;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_UNSUPPORTED);
    options = record<OmniUsdLiveRuntimeOptions>();
    options.flags = UINT32_MAX;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    options = record<OmniUsdLiveRuntimeOptions>();
    options.retry_initial_ms = 100;
    options.retry_maximum_ms = 1;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    options = record<OmniUsdLiveRuntimeOptions>();
    options.reconnect_initial_ms = 100;
    options.reconnect_maximum_ms = 1;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    options = record<OmniUsdLiveRuntimeOptions>();
    options.max_update_bytes = UINT32_MAX;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    // Explicit defaults are semantically equivalent, including a copied nested record.
    auto defaults = RuntimeOptions::copy(nullptr);
    // Unary retries and read or join reconnects have separate defaults.
    assert(defaults.values.retry_initial_ms == 100 && defaults.values.retry_maximum_ms == 10000 &&
           defaults.values.retry_max_attempts == 8);
    assert(defaults.values.reconnect_initial_ms == 1000 && defaults.values.reconnect_maximum_ms == 60000 &&
           defaults.values.reconnect_max_attempts == 10);
    {
        auto reconnect = record<OmniUsdLiveRuntimeOptions>();
        reconnect.reconnect_initial_ms = 7;
        reconnect.reconnect_maximum_ms = 70;
        reconnect.reconnect_max_attempts = 3;
        const auto copied = RuntimeOptions::copy(&reconnect);
        assert(copied.values.reconnect_initial_ms == 7 && copied.values.reconnect_maximum_ms == 70 &&
               copied.values.reconnect_max_attempts == 3 && copied.values.retry_initial_ms == 100 &&
               copied.values.retry_maximum_ms == 10000 && copied.values.retry_max_attempts == 8);
        assert(!(copied == defaults));
    }
    options = defaults.values;
    auto jitter = defaults.jitter;
    options.default_jitter = &jitter;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    jitter.delay_maximum_ms++;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(getRuntime()->options.jitter.delay_maximum_ms == defaults.jitter.delay_maximum_ms);
    assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_OK); // one of two owners
    finish();
}

void endpointOwnership()
{
    auto options = endpointOptions("LOCALHOST.:443");
    OmniUsdLiveEndpoint handle = 99;
    assert(omniUsdLiveRegisterEndpoint(&options, &handle) == OMNI_USD_LIVE_NOT_INITIALIZED && handle == 0);
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    char token[] = "synthetic-token";
    options.access_token = token;
    assert(omniUsdLiveRegisterEndpoint(&options, &handle) == OMNI_USD_LIVE_OK);
    token[0] = 'X'; // input ownership ends on return
    auto runtime = getRuntime();
    auto endpoint = runtime->findEndpoint(handle);
    assert(endpoint->token == "synthetic-token");
    assert(!endpoint->options.secure && endpoint->options.authority.canonical == "localhost");
    const auto writer = endpoint->writerId;
    options = endpointOptions("localhost");
    options.access_token = "stale-registration-token";
    OmniUsdLiveEndpoint duplicate = 0;
    assert(omniUsdLiveRegisterEndpoint(&options, &duplicate) == OMNI_USD_LIVE_OK && duplicate == handle);
    assert(endpoint->token == "synthetic-token" && endpoint->writerId == writer);
    options.transport_security = OMNI_USD_LIVE_TRANSPORT_TLS;
    duplicate = 99;
    assert(omniUsdLiveRegisterEndpoint(&options, &duplicate) == OMNI_USD_LIVE_ALREADY_EXISTS && duplicate == 0);
    auto dependency = runtime->acquireEndpoint(handle);
    assert(dependency);
    assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_BUSY);
    assert(omniUsdLiveUnregisterEndpoint(handle, 0) == OMNI_USD_LIVE_OK); // nonfinal owner
    assert(omniUsdLiveUnregisterEndpoint(handle, 0) == OMNI_USD_LIVE_BUSY);
    struct ExtendedStatus { OmniUsdLiveStatus status; uint64_t tail; } extended{};
    extended.status = record<OmniUsdLiveStatus>();
    extended.status.struct_size = sizeof(extended);
    extended.status.abi_version |= 5; // newer minor, same major
    extended.tail = UINT64_C(0x123456789abcdef0);
    assert(omniUsdLiveGetEndpointStatus(handle, &extended.status) == OMNI_USD_LIVE_OK);
    assert(extended.status.struct_size == sizeof(extended) && extended.tail == UINT64_C(0x123456789abcdef0));
    assert(extended.status.subject_id == handle && extended.status.state == OMNI_USD_LIVE_STATE_OPENING);
    {
        // A caller built against the 1.0 layout passes the prefix before
        // buffered_edit_count and receives exactly that prefix.
        constexpr uint32_t prefix = offsetof(OmniUsdLiveStatus, buffered_edit_count);
        struct OlderStatus { unsigned char bytes[prefix]; uint64_t tail; } older;
        std::memset(&older, 0xAB, sizeof(older));
        const uint32_t abi = OMNI_USD_LIVE_ABI_VERSION;
        std::memcpy(older.bytes, &prefix, sizeof(prefix)); std::memcpy(older.bytes + 4, &abi, sizeof(abi));
        assert(omniUsdLiveGetEndpointStatus(handle, reinterpret_cast<OmniUsdLiveStatus*>(&older)) == OMNI_USD_LIVE_OK);
        assert(older.tail == UINT64_C(0xABABABABABABABAB));
        uint64_t subject = 0; uint32_t size = 0;
        std::memcpy(&size, older.bytes, sizeof(size));
        std::memcpy(&subject, older.bytes + offsetof(OmniUsdLiveStatus, subject_id), sizeof(subject));
        assert(size == prefix && subject == handle);
        // A size that splits the appended field is rejected before any write.
        const uint32_t split = prefix + 4;
        std::memset(&older, 0xAB, sizeof(older));
        std::memcpy(older.bytes, &split, sizeof(split)); std::memcpy(older.bytes + 4, &abi, sizeof(abi));
        const auto before = older;
        assert(omniUsdLiveGetEndpointStatus(handle, reinterpret_cast<OmniUsdLiveStatus*>(&older)) ==
               OMNI_USD_LIVE_INVALID_ARGUMENT);
        assert(std::memcmp(&older, &before, sizeof(older)) == 0);
        const auto layer = makeLayerUri(makeSessionUri("localhost", "split-status"), "root.live");
        assert(omniUsdLiveGetLayerStatus(layer.c_str(), reinterpret_cast<OmniUsdLiveStatus*>(&older)) ==
               OMNI_USD_LIVE_INVALID_ARGUMENT);
        assert(std::memcmp(&older, &before, sizeof(older)) == 0);
        std::memcpy(older.bytes, &prefix, sizeof(prefix)); // The same unattached URI at the 1.0 size.
        assert(omniUsdLiveGetLayerStatus(layer.c_str(), reinterpret_cast<OmniUsdLiveStatus*>(&older)) ==
               OMNI_USD_LIVE_NOT_FOUND);
    }
    dependency = {};
    assert(omniUsdLiveUnregisterEndpoint(handle, 0) == OMNI_USD_LIVE_OK);
    assert(!runtime->acquireEndpoint(handle));
    assert(omniUsdLiveUnregisterEndpoint(handle, 0) == OMNI_USD_LIVE_INVALID_HANDLE);
    assert(omniUsdLiveGetEndpointStatus(handle, &extended.status) == OMNI_USD_LIVE_INVALID_HANDLE);
    auto fresh = add("localhost");
    assert(fresh != handle && runtime->findEndpoint(fresh)->writerId != writer);
    auto alias = add("127.0.0.1");
    assert(runtime->findEndpoint(alias)->writerId != runtime->findEndpoint(fresh)->writerId);
    assert(omniUsdLiveUnregisterEndpoint(fresh, 0) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterEndpoint(alias, 0) == OMNI_USD_LIVE_OK);
    // Exercise the last-reference closed-object path: its mutex must survive unlock.
    assert(!EndpointLease(std::move(endpoint)));
    runtime.reset();
    finish();
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    auto nextLifetime = add("localhost");
    assert(nextLifetime != fresh && getRuntime()->findEndpoint(nextLifetime)->writerId != writer);
    assert(omniUsdLiveUnregisterEndpoint(nextLifetime, 0) == OMNI_USD_LIVE_OK);
    finish();
}

void jitterConfiguration()
{
    auto params = record<OmniUsdLiveJitterParams>();
    assert(omniUsdLiveConfigureJitterReduction(0, nullptr) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(omniUsdLiveConfigureJitterReduction(0, &params) == OMNI_USD_LIVE_NOT_INITIALIZED);
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    auto first = add("localhost");
    auto retained = runtime->findEndpoint(first);
    assert(retained->jitter.delay_constant_ms == 10 && retained->jitter.delay_multiple == 2 &&
           retained->jitter.delay_maximum_ms == 1000);
    params.struct_size--;
    assert(omniUsdLiveConfigureJitterReduction(first, &params) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    params = record<OmniUsdLiveJitterParams>();
    params.abi_version = 2U << 16;
    assert(omniUsdLiveConfigureJitterReduction(first, &params) == OMNI_USD_LIVE_UNSUPPORTED);
    params = record<OmniUsdLiveJitterParams>();
    params.delay_constant_ms = UINT32_MAX;
    params.delay_multiple = UINT32_MAX;
    params.delay_maximum_ms = 0; // Zero disables holdback, including with huge other terms.
    assert(omniUsdLiveConfigureJitterReduction(0, &params) == OMNI_USD_LIVE_OK);
    params.delay_constant_ms = 17; // Configuration owns its copy.
    assert(runtime->defaultJitter.delay_constant_ms == UINT32_MAX);
    assert(runtime->options.jitter.delay_constant_ms == 10); // Initialize equivalence stays immutable.
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_OK);
    assert(retained->jitter.delay_maximum_ms == 1000); // Defaults affect new endpoints only.
    auto duplicate = add("LOCALHOST.");
    assert(duplicate == first && retained->jitter.delay_maximum_ms == 1000);
    auto second = add("127.0.0.1");
    auto inherited = runtime->findEndpoint(second);
    assert(inherited->jitter.delay_constant_ms == UINT32_MAX &&
           inherited->jitter.delay_multiple == UINT32_MAX && inherited->jitter.delay_maximum_ms == 0);
    struct Extended { OmniUsdLiveJitterParams params; uint64_t tail; } extended{};
    extended.params = record<OmniUsdLiveJitterParams>();
    extended.params.struct_size = sizeof(extended);
    extended.params.abi_version |= 7;
    extended.params.delay_constant_ms = 5;
    extended.params.delay_multiple = 3;
    extended.params.delay_maximum_ms = UINT32_MAX;
    extended.tail = UINT64_C(0x123456789abcdef0);
    assert(omniUsdLiveConfigureJitterReduction(first, &extended.params) == OMNI_USD_LIVE_OK);
    assert(extended.tail == UINT64_C(0x123456789abcdef0) && extended.params.struct_size == sizeof(extended));
    assert(retained->jitter.struct_size == sizeof(OmniUsdLiveJitterParams) &&
           retained->jitter.abi_version == OMNI_USD_LIVE_ABI_VERSION &&
           retained->jitter.delay_constant_ms == 5 && retained->jitter.delay_multiple == 3 &&
           retained->jitter.delay_maximum_ms == UINT32_MAX);
    assert(inherited->jitter.delay_maximum_ms == 0 && runtime->defaultJitter.delay_maximum_ms == 0);
    assert(omniUsdLiveConfigureJitterReduction(UINT64_MAX, &params) == OMNI_USD_LIVE_INVALID_HANDLE);
    assert(omniUsdLiveUnregisterEndpoint(duplicate, 0) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterEndpoint(first, 0) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveConfigureJitterReduction(first, &params) == OMNI_USD_LIVE_INVALID_HANDLE);
    assert(omniUsdLiveUnregisterEndpoint(second, 0) == OMNI_USD_LIVE_OK);
    inherited.reset();
    retained.reset();
    runtime.reset();
    finish();
}

void securityAndLimits()
{
    auto limits = record<OmniUsdLiveRuntimeOptions>();
    limits.max_status_subjects = 1;
    assert(omniUsdLiveInitialize(&limits) == OMNI_USD_LIVE_OK);
    auto options = endpointOptions("192.0.2.1");
    OmniUsdLiveEndpoint handle = 99;
    assert(omniUsdLiveRegisterEndpoint(&options, &handle) == OMNI_USD_LIVE_INVALID_ARGUMENT && handle == 0);
    options.tls_server_name = "[::1]";
    assert(omniUsdLiveRegisterEndpoint(&options, &handle) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    options.tls_server_name = "live.example:443";
    assert(omniUsdLiveRegisterEndpoint(&options, &handle) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    options.tls_server_name = "LIVE.EXAMPLE.";
    assert(omniUsdLiveRegisterEndpoint(&options, &handle) == OMNI_USD_LIVE_OK);
    assert(getRuntime()->findEndpoint(handle)->options.secure);
    auto conflicting = endpointOptions("live.example");
    conflicting.tls_server_name = "wrong.example";
    OmniUsdLiveEndpoint second = 99;
    assert(omniUsdLiveRegisterEndpoint(&conflicting, &second) == OMNI_USD_LIVE_INVALID_ARGUMENT && second == 0);
    conflicting.tls_server_name = nullptr;
    assert(omniUsdLiveRegisterEndpoint(&conflicting, &second) == OMNI_USD_LIVE_RESOURCE_EXHAUSTED && second == 0);
    auto duplicate = handle;
    assert(omniUsdLiveRegisterEndpoint(&options, &duplicate) == OMNI_USD_LIVE_OK && duplicate == handle);
    assert(omniUsdLiveUnregisterEndpoint(handle, 0) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterEndpoint(handle, 0) == OMNI_USD_LIVE_OK);
    auto local = add("[::1]:31415");
    auto status = record<OmniUsdLiveStatus>();
    assert(omniUsdLiveGetEndpointStatus(local, &status) == OMNI_USD_LIVE_OK);
    assert(status.state == OMNI_USD_LIVE_STATE_AUTH_WAIT && status.result == OMNI_USD_LIVE_UNAUTHENTICATED);
    assert(!getRuntime()->findEndpoint(local)->options.secure);
    assert(omniUsdLiveUnregisterEndpoint(local, 0) == OMNI_USD_LIVE_OK);
    finish();
}

void shortStructuresAndEquivalentTls()
{
#if defined(__unix__)
    const auto page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    auto* memory = static_cast<char*>(mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE,
                                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    assert(memory != MAP_FAILED);
    assert(mprotect(memory + page, page, PROT_NONE) == 0);
    auto* sizeOnly = reinterpret_cast<uint32_t*>(memory + page - sizeof(uint32_t));
    *sizeOnly = sizeof(uint32_t);
    assert(omniUsdLiveInitialize(reinterpret_cast<OmniUsdLiveRuntimeOptions*>(sizeOnly)) ==
           OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(omniUsdLiveGetCapabilities(reinterpret_cast<OmniUsdLiveCapabilities*>(sizeOnly)) ==
           OMNI_USD_LIVE_INVALID_ARGUMENT);
    auto runtimeOptions = record<OmniUsdLiveRuntimeOptions>();
    runtimeOptions.default_jitter = reinterpret_cast<OmniUsdLiveJitterParams*>(sizeOnly);
    assert(omniUsdLiveInitialize(&runtimeOptions) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    {
        // A caller built before apply_lag_warning_threshold passes the prefix
        // that ends before it, here against the guard page: the runtime reads
        // nothing past it and selects the default, equal to NULL options.
        constexpr uint32_t prefix = offsetof(OmniUsdLiveRuntimeOptions, apply_lag_warning_threshold);
        static_assert(prefix % alignof(OmniUsdLiveRuntimeOptions) == 0);
        auto defaults = record<OmniUsdLiveRuntimeOptions>();
        defaults.diagnostic_min_severity = OMNI_USD_LIVE_SEVERITY_INFO;
        defaults.struct_size = prefix;
        auto* older = memory + page - prefix;
        std::memcpy(older, &defaults, prefix);
        assert(omniUsdLiveInitialize(reinterpret_cast<OmniUsdLiveRuntimeOptions*>(older)) == OMNI_USD_LIVE_OK);
        assert(getRuntime()->options.values.apply_lag_warning_threshold == 256);
        assert(getRuntime()->options.values.reconnect_initial_ms == 1000);
        assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_OK); // One of two owners.
        // A size that splits the appended field is rejected.
        defaults.struct_size = prefix + 4;
        assert(omniUsdLiveInitialize(&defaults) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    }
    {
        // A caller built before the reconnect options passes the prefix that
        // ends before them: it keeps apply_lag_warning_threshold and selects
        // the reconnect defaults, equal to NULL options.
        constexpr uint32_t prefix = offsetof(OmniUsdLiveRuntimeOptions, reconnect_initial_ms);
        auto defaults = record<OmniUsdLiveRuntimeOptions>();
        defaults.diagnostic_min_severity = OMNI_USD_LIVE_SEVERITY_INFO;
        defaults.struct_size = prefix;
        auto* older = memory + page - prefix;
        std::memcpy(older, &defaults, prefix);
        assert(omniUsdLiveInitialize(reinterpret_cast<OmniUsdLiveRuntimeOptions*>(older)) == OMNI_USD_LIVE_OK);
        const auto& values = getRuntime()->options.values;
        assert(values.reconnect_initial_ms == 1000 && values.reconnect_maximum_ms == 60000 &&
               values.reconnect_max_attempts == 10 && values.apply_lag_warning_threshold == 256);
        assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_OK); // One of two owners.
        // The reconnect options are appended together: a size inside them is rejected.
        for (uint32_t split : {prefix + 4, prefix + 8, prefix + 12}) {
            defaults.struct_size = split;
            assert(omniUsdLiveInitialize(&defaults) == OMNI_USD_LIVE_INVALID_ARGUMENT);
        }
    }
    OmniUsdLiveEndpoint output = 99;
    assert(omniUsdLiveRegisterEndpoint(reinterpret_cast<OmniUsdLiveEndpointOptions*>(sizeOnly), &output) ==
           OMNI_USD_LIVE_INVALID_ARGUMENT && output == 0);
    assert(omniUsdLiveGetEndpointStatus(0, reinterpret_cast<OmniUsdLiveStatus*>(sizeOnly)) ==
           OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(omniUsdLiveConfigureJitterReduction(0, reinterpret_cast<OmniUsdLiveJitterParams*>(sizeOnly)) ==
           OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(munmap(memory, 2 * page) == 0);
    finish();
#endif
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    for (unsigned order = 0; order < 2; ++order) {
        auto first = endpointOptions("LIVE.EXAMPLE.:443");
        auto second = endpointOptions("live.example");
        (order ? first : second).tls_server_name = "Live.Example.";
        OmniUsdLiveEndpoint a = 0, b = 0;
        assert(omniUsdLiveRegisterEndpoint(&first, &a) == OMNI_USD_LIVE_OK);
        assert(omniUsdLiveRegisterEndpoint(&second, &b) == OMNI_USD_LIVE_OK && a == b);
        assert(omniUsdLiveUnregisterEndpoint(a, 0) == OMNI_USD_LIVE_OK);
        assert(omniUsdLiveUnregisterEndpoint(b, 0) == OMNI_USD_LIVE_OK);
    }
    finish();
}

class Barrier {
public:
    explicit Barrier(unsigned parties) : parties_(parties) {}
    void arrive()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto generation = generation_;
        if (++arrived_ == parties_) {
            arrived_ = 0;
            ++generation_;
            ready_.notify_all();
        } else {
            ready_.wait(lock, [&] { return generation_ != generation; });
        }
    }
private:
    const unsigned parties_;
    unsigned arrived_ = 0;
    unsigned generation_ = 0;
    std::mutex mutex_;
    std::condition_variable ready_;
};

void concurrentOwners()
{
    constexpr unsigned workers = 8;
    constexpr unsigned rounds = 80;
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    Barrier barrier(workers);
    std::vector<OmniUsdLiveEndpoint> handles(workers);
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < workers; ++i) {
        threads.emplace_back([&, i] {
            for (unsigned round = 0; round < rounds; ++round) {
                barrier.arrive();
                assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
                handles[i] = add(i % 2 ? "LOCALHOST.:443" : "localhost");
                auto dependency = runtime->acquireEndpoint(handles[i]);
                assert(dependency);
                barrier.arrive();
                for (auto handle : handles)
                    assert(handle == handles[i]);
                auto status = record<OmniUsdLiveStatus>();
                assert(omniUsdLiveGetEndpointStatus(handles[i], &status) == OMNI_USD_LIVE_OK);
                // No worker retains a lease once final unregister becomes possible.
                dependency = {};
                barrier.arrive();
                assert(omniUsdLiveUnregisterEndpoint(handles[i], 0) == OMNI_USD_LIVE_OK);
                assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_OK); // root retains one owner
            }
        });
    }
    for (auto& thread : threads)
        thread.join();
    assert(runtime->endpoints.empty());
    runtime.reset();
    finish();
}

struct AlarmTag final : CompletionTag {
    explicit AlarmTag(EndpointLease value) : lease(std::move(value)) {}
    grpc::Alarm alarm;
    EndpointLease lease;
    std::promise<bool> done;
    void onComplete(bool ok, Clock::time_point) noexcept override
    {
        lease = {}; // endpoint remains owned until completion consumes its state
        done.set_value(ok);
    }
};

void autonomousCompletion()
{
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    auto handle = add("localhost");
    auto runtime = getRuntime();
    std::vector<std::future<bool>> completions;
    for (unsigned i = 0; i < 100; ++i) {
        auto tag = std::make_unique<AlarmTag>(runtime->acquireEndpoint(handle));
        completions.push_back(tag->done.get_future());
        auto* submitted = submitTag(tag, *runtime->diagnostics);
        submitted->alarm.Set(&runtime->queue(), std::chrono::system_clock::now(), submitted);
    }
    // No frame pump or caller-thread processing. The runtime's native CQ owns tags.
    for (auto& completion : completions) {
        assert(completion.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
        assert(completion.get());
    }
    assert(omniUsdLiveUnregisterEndpoint(handle, 0) == OMNI_USD_LIVE_OK);
    finish();
    const auto tags = runtime->diagnostics->snapshot(0, 0).metrics.cqTags;
    assert(tags.submitted == 100 && tags.receivedOk == 100 && tags.retired == 100);
    assert(tags.outstanding == 0 && tags.handling == 0 && tags.retained == 0 && tags.abandoned == 0);
    runtime.reset();
}

void unregisterAgainstAdmission()
{
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    for (unsigned round = 0; round < 100; ++round) {
        const auto handle = add("localhost");
        Barrier barrier(2);
        bool admitted = false;
        OmniUsdLiveResult removal = OMNI_USD_LIVE_INTERNAL_ERROR;
        std::thread acquire([&] {
            barrier.arrive();
            auto lease = runtime->acquireEndpoint(handle);
            admitted = bool(lease);
            barrier.arrive(); // Keep the lease until the removal attempt finishes.
        });
        std::thread remove([&] {
            barrier.arrive();
            removal = omniUsdLiveUnregisterEndpoint(handle, 0);
            barrier.arrive();
        });
        acquire.join();
        remove.join();
        assert(admitted ? removal == OMNI_USD_LIVE_BUSY : removal == OMNI_USD_LIVE_OK);
        if (admitted)
            assert(omniUsdLiveUnregisterEndpoint(handle, 0) == OMNI_USD_LIVE_OK);
        assert(!runtime->acquireEndpoint(handle));
    }
    runtime.reset();
    finish();
}

struct HeldCompletion {
    std::promise<void> entered;
    std::promise<void> release;
    std::promise<void> destructorEntered, destructorRelease;
    std::atomic<unsigned> destroyed{0};
};

// Deliberately stall a test-only tag. Product CQ handlers never wait on hosts;
// this fixture forces a pending shutdown instead of depending on OS timing.
struct HeldTag final : CompletionTag {
    explicit HeldTag(std::shared_ptr<HeldCompletion> value)
        : state(std::move(value)), release(state->release.get_future()) {}
    ~HeldTag() override {
        state->destructorEntered.set_value();
        assert(state->destructorRelease.get_future().wait_for(std::chrono::seconds(3)) == std::future_status::ready);
        ++state->destroyed;
    }
    grpc::Alarm alarm;
    std::shared_ptr<HeldCompletion> state;
    std::future<void> release;
    void onComplete(bool ok, Clock::time_point) noexcept override
    {
        assert(ok);
        state->entered.set_value();
        release.wait();
    }
};

void shutdownTimeoutRetainsOwnership()
{
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    auto state = std::make_shared<HeldCompletion>();
    auto entered = state->entered.get_future();
    auto tag = std::make_unique<HeldTag>(state);
    auto* submitted = submitTag(tag, *runtime->diagnostics);
    submitted->alarm.Set(&runtime->queue(), std::chrono::system_clock::now(), submitted);
    assert(entered.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    auto tags = runtime->diagnostics->snapshot(0, 0).metrics.cqTags;
    assert(tags.submitted == 1 && tags.receivedOk == 1 && tags.outstanding == 0);
    assert(tags.handling == 1 && tags.retained == 1 && tags.retired == 0);
    assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_TIMEOUT);
    assert(getRuntime() == runtime && runtime->closing);
    {
        // Diagnostics remain queryable after sink delivery has stopped.
        uint64_t size=0;
        assert(omniUsdLiveCopyDiagnostics(0,nullptr,nullptr,&size)==OMNI_USD_LIVE_OK && size>1);
        std::vector<char> bytes(static_cast<size_t>(size)+4096); size=bytes.size();
        assert(omniUsdLiveCopyDiagnostics(0,nullptr,bytes.data(),&size)==OMNI_USD_LIVE_OK);
        const std::string json(bytes.data());
        assert(json.find("\"event_name\":\"quiescence_failed\"")!=std::string::npos);
        assert(json.find("\"shutdown_duration_ms_by_phase_result\":")!=std::string::npos);
        assert(json.find("\"phase\":2,\"result\":9,\"count\":1")!=std::string::npos);
    }
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_BUSY);
    auto options = endpointOptions("localhost");
    OmniUsdLiveEndpoint endpoint = 99;
    assert(omniUsdLiveRegisterEndpoint(&options, &endpoint) == OMNI_USD_LIVE_BUSY && endpoint == 0);
    assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_TIMEOUT);
    state->release.set_value();
    assert(state->destructorEntered.get_future().wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    tags = runtime->diagnostics->snapshot(0, 0).metrics.cqTags;
    assert(tags.handling == 0 && tags.retained == 1 && tags.retired == 0);
    assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_TIMEOUT);
    state->destructorRelease.set_value();
    assert(omniUsdLiveShutdown(2000) == OMNI_USD_LIVE_OK);
    assert(state->destroyed == 1 && !getRuntime());
    const auto observed=runtime->diagnostics->snapshot(0,100);
    const auto& attempts=observed.metrics.shutdownDurationByPhaseResult;
    assert(attempts[static_cast<size_t>(ShutdownPhase::CompletionQueue)][OMNI_USD_LIVE_TIMEOUT].count==3);
    assert(attempts[static_cast<size_t>(ShutdownPhase::Complete)][OMNI_USD_LIVE_OK].count==1);
    assert(observed.metrics.events[OMNI_USD_LIVE_RECORD_QUIESCENCE_FAILED]==3);
    size_t failed=0;
    for(const auto& entry:observed.records) if(entry.eventCode==OMNI_USD_LIVE_RECORD_QUIESCENCE_FAILED) {
        ++failed;
        assert(entry.endpoint==0 && entry.subjectId==0 && entry.incidentId==0 && entry.resource[0]==0);
        assert(entry.severity==OMNI_USD_LIVE_SEVERITY_ERROR && entry.fieldCount==4);
        assert(entry.fields[0].key==DiagnosticKey::Operation && entry.fields[0].value==OMNI_USD_LIVE_OPERATION_SHUTDOWN);
        assert(entry.fields[1].key==DiagnosticKey::Result && entry.fields[1].value==OMNI_USD_LIVE_TIMEOUT);
        assert(entry.fields[2].key==DiagnosticKey::DurationMs);
        assert(entry.fields[3].key==DiagnosticKey::Reason && entry.fields[3].value==static_cast<uint32_t>(ShutdownPhase::CompletionQueue));
    }
    assert(failed==3);
    tags = runtime->diagnostics->snapshot(0, 0).metrics.cqTags;
    assert(tags.retained == 0 && tags.retired == 1 && tags.abandoned == 0);
    assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_NOT_INITIALIZED);
    runtime.reset();
}
void shutdownReferenceAndAdmissionMetrics()
{
    assert(omniUsdLiveInitialize(nullptr)==OMNI_USD_LIVE_OK);
    auto original=getRuntime();
    assert(omniUsdLiveInitialize(nullptr)==OMNI_USD_LIVE_OK);
    assert(omniUsdLiveShutdown(0)==OMNI_USD_LIVE_OK && !original->closing);
    const auto endpoint=add("localhost:35719");
    assert(omniUsdLiveShutdown(0)==OMNI_USD_LIVE_BUSY && !original->closing);
    auto metrics=original->diagnostics->snapshot(0,0).metrics;
    assert(metrics.shutdownDurationByPhaseResult[0][OMNI_USD_LIVE_OK].count==1);
    assert(metrics.shutdownDurationByPhaseResult[1][OMNI_USD_LIVE_BUSY].count==1);
    assert(metrics.events[OMNI_USD_LIVE_RECORD_QUIESCENCE_FAILED]==0);
    assert(omniUsdLiveUnregisterEndpoint(endpoint,0)==OMNI_USD_LIVE_OK);
    assert(omniUsdLiveShutdown(3000)==OMNI_USD_LIVE_OK);
    assert(omniUsdLiveInitialize(nullptr)==OMNI_USD_LIVE_OK);
    const auto replacement=getRuntime();
    assert(replacement!=original);
    assert(original->diagnostics->snapshot(0,0).metrics.shutdownDurationByPhaseResult[8][OMNI_USD_LIVE_OK].count==1);
    metrics=replacement->diagnostics->snapshot(0,0).metrics;
    for(const auto& phase:metrics.shutdownDurationByPhaseResult)
        for(const auto& result:phase) assert(result.count==0);
    assert(omniUsdLiveShutdown(3000)==OMNI_USD_LIVE_OK);
}
void pendingByteBudget()
{
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    auto options = endpointOptions("budget.example.invalid");
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&options, &endpoint) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    auto retained = runtime->findEndpoint(endpoint);
    std::promise<void> release;
    auto gate = release.get_future().share();
    std::atomic<unsigned> ready{0}, admitted{0}, rejected{0};
    std::vector<std::unique_ptr<PendingBytes>> reservations(16);
    std::vector<std::thread> writers;
    for (unsigned index = 0; index < reservations.size(); ++index) {
        writers.emplace_back([&, index] {
            ++ready;
            gate.wait();
            try {
                reservations[index] = std::make_unique<PendingBytes>(runtime->acquireEndpoint(endpoint), 64, 256);
                ++admitted;
            } catch (OmniUsdLiveResult result) {
                assert(result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
                ++rejected;
            }
        });
    }
    while (ready != reservations.size()) std::this_thread::yield();
    release.set_value();
    for (auto& writer : writers) writer.join();
    assert(admitted == 4 && rejected == 12);
    assert(retained->pendingTransportBytes == 256);
    auto metrics = runtime->diagnostics->snapshot(0, 0).metrics.pendingBytes;
    assert(metrics.reservations.admitted == 4 && metrics.reservations.rejected == 12 && metrics.reservations.current == 4);
    assert(metrics.reservations.highWater == 4 && metrics.admittedBytes == 256 && metrics.rejectedBytes == 768);
    assert(metrics.currentBytes.low == 256 && metrics.currentBytes.high == 0 && metrics.highWaterBytes.low == 256);
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_BUSY);
    reservations.clear();
    assert(retained->pendingTransportBytes == 0);
    metrics = runtime->diagnostics->snapshot(0, 0).metrics.pendingBytes;
    assert(metrics.reservations.current == 0 && metrics.reservations.released == 4 && metrics.releasedBytes == 256);
    assert(metrics.currentBytes.low == 0 && metrics.currentBytes.high == 0);
    try {
        PendingBytes overflow(runtime->acquireEndpoint(endpoint), UINT64_MAX, 256);
        assert(false);
    } catch (OmniUsdLiveResult result) { assert(result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED); }
    assert(retained->pendingTransportBytes == 0);
    {
        PendingBytes first(runtime->acquireEndpoint(endpoint), 256, 256);
        PendingBytes moved(std::move(first));
        assert(retained->pendingTransportBytes == 256);
        metrics = runtime->diagnostics->snapshot(0, 0).metrics.pendingBytes;
        assert(metrics.reservations.admitted == 5 && metrics.reservations.current == 1 && metrics.reservations.released == 4);
    }
    assert(retained->pendingTransportBytes == 0);
    {
        PendingBytes zero(runtime->acquireEndpoint(endpoint), 0, 256);
        PendingBytes moved(std::move(zero));
        metrics = runtime->diagnostics->snapshot(0, 0).metrics.pendingBytes;
        assert(metrics.reservations.current == 1 && metrics.currentBytes.low == 0);
    }
    metrics = runtime->diagnostics->snapshot(0, 0).metrics.pendingBytes;
    assert(metrics.reservations.admitted == 6 && metrics.reservations.released == 6 && metrics.reservations.current == 0);
    assert(metrics.reservations.rejected == 13 && metrics.rejectedBytes == UINT64_MAX);
    assert(metrics.admittedBytes == 512 && metrics.releasedBytes == 512 && metrics.highWaterBytes.low == 256);
    {
        // Every release reports to each live watcher on its endpoint; only a
        // watcher whose flush a transient cause deferred requests a retry.
        auto healthy = std::make_shared<BufferedEditCount>(), blocked = std::make_shared<BufferedEditCount>();
        unsigned healthyRetries = 0, blockedRetries = 0;
        healthy->setRetry([&] { ++healthyRetries; });
        blocked->setRetry([&] {
            ++blockedRetries;
            assert(retained->pendingTransportBytes == 0); // Runs after the release, outside the endpoint lock.
        });
        (void)blocked->note(1, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        {
            std::lock_guard<std::mutex> lock(retained->mutex);
            retained->pendingByteWatchers.push_back(healthy); retained->pendingByteWatchers.push_back(blocked);
        }
        { PendingBytes held(runtime->acquireEndpoint(endpoint), 64, 256); assert(!blockedRetries && !healthyRetries); }
        assert(blockedRetries == 1 && !healthyRetries && healthy->releases() == 1 && blocked->releases() == 1);
        (void)blocked->note(0); // An admission clears the deferral cause.
        { PendingBytes held(runtime->acquireEndpoint(endpoint), 64, 256); }
        assert(blockedRetries == 1 && blocked->releases() == 2);
        healthy.reset();
        { PendingBytes held(runtime->acquireEndpoint(endpoint), 64, 256); }
        assert(retained->pendingByteWatchers.size() == 1 && blocked->releases() == 3); // Expired watchers are pruned.
        blocked.reset();
        { PendingBytes held(runtime->acquireEndpoint(endpoint), 64, 256); }
        assert(retained->pendingByteWatchers.empty());
    }
    {
        // Every layer's host edit buffer on an endpoint shares one budget,
        // separate from transport reservations: charges always apply, the
        // budget reports exhausted once all of them together reach the limit,
        // a transport reservation never counts them, a decrease reports to
        // the watchers and destruction releases the charge.
        auto blocked = std::make_shared<BufferedEditCount>();
        unsigned retries = 0;
        blocked->setRetry([&] { ++retries; });
        (void)blocked->note(0, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        {
            std::lock_guard<std::mutex> lock(retained->mutex);
            retained->pendingByteWatchers.push_back(blocked);
        }
        {
            HostBufferCharge first(runtime->acquireEndpoint(endpoint), 256), second(runtime->acquireEndpoint(endpoint), 256);
            first.set(200);
            assert(!first.budget().exhausted && !second.budget().exhausted && first.budget().charged == 200);
            second.set(100); // A started turn always records, past the budget.
            const auto budget = second.budget();
            assert(budget.exhausted && first.budget().exhausted && budget.charged == 300 && budget.limit == 256);
            assert(retained->hostBufferBytes == 300 && retained->pendingTransportBytes == 0);
            { PendingBytes transport(runtime->acquireEndpoint(endpoint), 256, 256); } // Buffers never refuse it.
            const auto releases = blocked->releases();
            first.set(100);
            assert(!second.budget().exhausted && blocked->releases() == releases + 1 && retries == 2);
            second.set(150); // An increase reports nothing.
            assert(blocked->releases() == releases + 1 && retained->hostBufferBytes == 250);
        }
        assert(retained->hostBufferBytes == 0 && blocked->releases() >= 3);
        blocked.reset();
        { PendingBytes held(runtime->acquireEndpoint(endpoint), 64, 256); }
        assert(retained->pendingByteWatchers.empty());
    }
    retained.reset();
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
    runtime.reset();
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
}
} // namespace

void statusApiOwnership()
{
    struct Observed {
        std::mutex mutex;
        std::condition_variable changed;
        std::vector<OmniUsdLiveStatus> events;
        static void callback(void* user, const OmniUsdLiveStatusEvent* event) {
            auto& self = *static_cast<Observed*>(user);
            assert(event->endpoint && event->status->subject_kind == OMNI_USD_LIVE_SUBJECT_ENDPOINT);
            std::lock_guard<std::mutex> lock(self.mutex);
            self.events.push_back(*event->status); self.changed.notify_all();
        }
        void wait(uint32_t state) {
            std::unique_lock<std::mutex> lock(mutex);
            assert(changed.wait_for(lock, std::chrono::seconds(3), [&] {
                return !events.empty() && events.back().state == state;
            }));
        }
    } observed;
    auto status = record<OmniUsdLiveStatus>();
    assert(omniUsdLiveGetLayerStatus("live://status.invalid/session/root.live", &status) == OMNI_USD_LIVE_NOT_INITIALIZED);
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    OmniUsdLiveCallbackHandle global = 0, scoped = 0;
    assert(omniUsdLiveRegisterStatusCallback(0, &observed, Observed::callback, &global) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_BUSY);
    auto options = record<OmniUsdLiveEndpointOptions>();
    options.authority = "status.invalid"; options.access_token = "synthetic-status";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&options, &endpoint) == OMNI_USD_LIVE_OK);
    observed.wait(OMNI_USD_LIVE_STATE_OPENING);
    assert(omniUsdLiveGetEndpointStatus(endpoint, &status) == OMNI_USD_LIVE_OK);
    assert(status.subject_id == endpoint && status.revision && !status.pending_write_count);
    assert(omniUsdLiveGetLayerStatus("live://status.invalid/session/root.live", &status) == OMNI_USD_LIVE_NOT_FOUND);
    assert(omniUsdLiveGetLayerStatus("live://status.invalid/session", &status) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(omniUsdLiveRegisterStatusCallback(endpoint, nullptr, [](void*, const OmniUsdLiveStatusEvent*) {}, &scoped) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_BUSY);
    assert(omniUsdLiveUnregisterCallback(scoped, 3000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterCallback(scoped, 0) == OMNI_USD_LIVE_INVALID_HANDLE);
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
    observed.wait(OMNI_USD_LIVE_STATE_CLOSED);
    assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_BUSY);
    assert(omniUsdLiveUnregisterCallback(global, 3000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
}

void pullStatusApiOwnership()
{
    OmniUsdLiveCallbackHandle global = 91, scoped = 0;
    assert(omniUsdLiveCreateStatusObserver(0, nullptr) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(omniUsdLiveCreateStatusObserver(0, &global) == OMNI_USD_LIVE_NOT_INITIALIZED && global == 0);
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveCreateStatusObserver(0, &global) == OMNI_USD_LIVE_OK && (global >> 56) == 5);
    assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_BUSY);
    const auto endpoint = add("pull-status.invalid");
    assert(omniUsdLiveCreateStatusObserver(endpoint, &scoped) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_BUSY);
    auto output = record<OmniUsdLiveStatusObservation>();
    output.endpoint = 37;
    uint64_t size = 991;
    output.struct_size--;
    auto before = output;
    assert(omniUsdLiveReadStatus(global, 0, &output, nullptr, &size) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(size == 991 && std::memcmp(&output, &before, sizeof(output)) == 0);
    output = record<OmniUsdLiveStatusObservation>(); output.abi_version = 2u << 16; before = output;
    assert(omniUsdLiveReadStatus(global, 0, &output, nullptr, &size) == OMNI_USD_LIVE_UNSUPPORTED);
    assert(size == 991 && std::memcmp(&output, &before, sizeof(output)) == 0);
    output = record<OmniUsdLiveStatusObservation>(); before = output;
    assert(omniUsdLiveReadStatus(global, 0, &output, nullptr, nullptr) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(omniUsdLiveReadStatus(global, 0, &output, nullptr, &size) == OMNI_USD_LIVE_OK);
    assert(size == 1 && std::memcmp(&output, &before, sizeof(output)) == 0);
    struct Extended { OmniUsdLiveStatusObservation known; uint64_t trailing = UINT64_C(0x1234abcdef567890); } extended;
    extended.known = record<OmniUsdLiveStatusObservation>(); extended.known.struct_size = sizeof(extended);
    extended.known.abi_version |= 0xffff;
    char uri = '!'; size = 1;
    assert(omniUsdLiveReadStatus(global, 0, &extended.known, &uri, &size) == OMNI_USD_LIVE_OK);
    assert(uri == '\0' && size == 1 && extended.known.struct_size == sizeof(extended));
    assert(extended.trailing == UINT64_C(0x1234abcdef567890));
    assert(extended.known.endpoint == endpoint && extended.known.session == 0);
    assert(extended.known.status.subject_id == endpoint && extended.known.status.subject_kind == OMNI_USD_LIVE_SUBJECT_ENDPOINT);
    assert(extended.known.status.struct_size == sizeof(OmniUsdLiveStatus));
    {
        // A 1.0 observation ends before the inline buffered_edit_count.
        constexpr uint32_t prefix = offsetof(OmniUsdLiveStatusObservation, status) +
                                    offsetof(OmniUsdLiveStatus, buffered_edit_count);
        struct OlderObservation { unsigned char bytes[prefix]; uint64_t tail; } older;
        std::memset(&older, 0xAB, sizeof(older));
        const uint32_t abi = OMNI_USD_LIVE_ABI_VERSION;
        std::memcpy(older.bytes, &prefix, sizeof(prefix)); std::memcpy(older.bytes + 4, &abi, sizeof(abi));
        char name = '!'; size = 1;
        // The unacknowledged endpoint observation is read again.
        assert(omniUsdLiveReadStatus(global, 0, reinterpret_cast<OmniUsdLiveStatusObservation*>(&older), &name, &size) ==
               OMNI_USD_LIVE_OK);
        assert(older.tail == UINT64_C(0xABABABABABABABAB) && name == '\0');
        uint64_t subject = 0; uint32_t inner = 0;
        constexpr size_t status = offsetof(OmniUsdLiveStatusObservation, status);
        std::memcpy(&inner, older.bytes + status, sizeof(inner));
        std::memcpy(&subject, older.bytes + status + offsetof(OmniUsdLiveStatus, subject_id), sizeof(subject));
        assert(inner == offsetof(OmniUsdLiveStatus, buffered_edit_count) && subject == endpoint);
        const uint32_t split = prefix + 4;
        std::memset(&older, 0xAB, sizeof(older));
        std::memcpy(older.bytes, &split, sizeof(split)); std::memcpy(older.bytes + 4, &abi, sizeof(abi));
        assert(omniUsdLiveReadStatus(global, 0, reinterpret_cast<OmniUsdLiveStatusObservation*>(&older), &name, &size) ==
               OMNI_USD_LIVE_INVALID_ARGUMENT);
    }
    assert(omniUsdLiveAcknowledgeStatus(global, 0, extended.known.status.revision) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(omniUsdLiveAcknowledgeStatus(global, endpoint, 0) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(omniUsdLiveAcknowledgeStatus(global, endpoint, extended.known.status.revision) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterCallback(scoped, 3000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
    size = 1;
    assert(omniUsdLiveReadStatus(global, 3000, &output, &uri, &size) == OMNI_USD_LIVE_OK);
    assert(output.status.subject_id == endpoint && output.status.state == OMNI_USD_LIVE_STATE_CLOSED);
    assert(omniUsdLiveAcknowledgeStatus(global, endpoint, output.status.revision) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterCallback(global, 3000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveReadStatus(global, 0, &output, nullptr, &size) == OMNI_USD_LIVE_INVALID_HANDLE);
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
}

void testRequestLifecycle();
void testAuthController();
void testLayerWriteLifecycle();
void testLayerReadLifecycle();
void testLayerProgress();
void jitterEstimatorTests();
void drainTrackerTests();
void statusRegistryTests();
void diagnosticsRegistryTests();
void diagnosticsApiTests();
void testAuthDiagnostics();
void sessionPeersTests();
void sessionApiTests();
void layerAdmissionTests();
void layerStatusTests();

int main()
{
    assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_NOT_INITIALIZED);
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    rpcMonotonicDeadline();
    metadataDeadlineInheritance();
    assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveWaitForPendingUpdates(1000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
    staticCapabilities();
    abiAndConfiguration();
    jitterConfiguration();
    endpointOwnership();
    securityAndLimits();
    shortStructuresAndEquivalentTls();
    concurrentOwners();
    autonomousCompletion();
    unregisterAgainstAdmission();
    shutdownTimeoutRetainsOwnership();
    shutdownReferenceAndAdmissionMetrics();
    testRequestLifecycle();
    testAuthController();
    pendingByteBudget();
    testLayerWriteLifecycle();
    testLayerReadLifecycle();
    testLayerProgress();
    jitterEstimatorTests();
    drainTrackerTests();
    statusRegistryTests();
    diagnosticsRegistryTests();
    diagnosticsApiTests();
    testAuthDiagnostics();
    sessionPeersTests();
    sessionApiTests();
    layerAdmissionTests();
    layerStatusTests();
    statusApiOwnership();
    pullStatusApiOwnership();
    std::cout << "runtime foundation: ABI, ownership, configuration, 8 native owners/80 rounds, 100 CQ tags passed\n";
}
