// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Generates the LiveService client from the public contract file, the same
//! `live_service.proto` clients are given. Client code only: the suite never
//! serves the API, and it never shares the service's own generated code (the
//! service's `build.rs` configures its build differently, for example
//! `bytes::Bytes` subscribe chunks).

const PROTO: &str = "../proto/nvidia/omniverse/live/v1alpha/live_service.proto";

fn main() {
    // Rebuild when the contract changes or a different `protoc` is selected.
    println!("cargo:rerun-if-changed={PROTO}");
    println!("cargo:rerun-if-env-changed=PROTOC");

    tonic_prost_build::configure()
        .build_server(false)
        .build_client(true)
        .compile_protos(&[PROTO], &["../proto"])
        .expect("failed to compile the public LiveService proto");
}
