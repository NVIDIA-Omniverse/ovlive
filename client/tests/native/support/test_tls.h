// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Test-only TLS for drivers that dial a TLS-terminating fixture proxy:
// LIVE_TEST_TLS_CA names a PEM file of trusted roots and LIVE_TEST_TLS_NAME
// the certificate's DNS name. With both unset an endpoint keeps its default
// transport; exactly one set aborts, so a fixture never falls back to plaintext.
#include "omni/usd/live/Live.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
struct TestTls {
    std::string ca, name;
    TestTls() {
        const char* path = std::getenv("LIVE_TEST_TLS_CA");
        const char* host = std::getenv("LIVE_TEST_TLS_NAME");
        if (!path != !host) {
            std::cerr << "LIVE_TEST_TLS_CA and LIVE_TEST_TLS_NAME must be set together\n";
            std::abort();
        }
        if (!path) return;
        std::ifstream input(path, std::ios::binary);
        ca.assign(std::istreambuf_iterator<char>(input), {});
        if (ca.empty()) { std::cerr << "LIVE_TEST_TLS_CA is unreadable or empty\n"; std::abort(); }
        name = host;
    }
    void apply(OmniUsdLiveEndpointOptions& options) const {
        if (ca.empty()) return;
        options.transport_security = OMNI_USD_LIVE_TRANSPORT_TLS;
        options.tls_server_name = name.c_str();
        options.ca_pem = ca.c_str();
    }
};
