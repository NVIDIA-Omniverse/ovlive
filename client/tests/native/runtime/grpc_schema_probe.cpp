// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "nvidia/omniverse/live/v1alpha/live.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <cassert>

int main()
{
    using namespace nvidia::omniverse::live::v1alpha;
    assert(grpc::Version() == "1.65.4");
    auto channel = grpc::CreateChannel("127.0.0.1:1", grpc::InsecureChannelCredentials());
    auto stub = LiveService::NewStub(channel);
    assert(stub);
    CreateSessionResponse created;
    created.set_session_id("opaque/session");
    std::string bytes;
    assert(created.SerializeToString(&bytes));
    CreateSessionResponse decoded;
    assert(decoded.ParseFromString(bytes));
    assert(decoded.session_id() == created.session_id());
    assert(created.GetDescriptor()->field_count() == 1);
    assert(JoinSessionHeader::descriptor()->FindFieldByName("user_name") == nullptr);
    return 0;
}
