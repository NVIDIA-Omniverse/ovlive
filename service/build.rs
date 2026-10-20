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

use std::env;
use std::path::PathBuf;

fn main() {
    // Cargo normally handles `OUT_DIR` rebuilds, and `tonic-build` is
    // supposed to emit `rerun-if-env-changed=PROTOC` implicitly. Emit both
    // explicitly here (b.3z7 low) so an environment flip (e.g. a user
    // installs a newer `protoc` on PATH) triggers a rebuild without a
    // `cargo clean`.
    println!("cargo:rerun-if-env-changed=PROTOC");
    println!("cargo:rerun-if-env-changed=OUT_DIR");

    // ── Proto / tonic-build ──────────────────────────────────────────────────
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let descriptor_path = out_dir.join("live_service_descriptor.bin");

    // tonic 0.14 split code generation into a transport-agnostic `tonic-build`
    // crate and backend-specific helpers. `tonic-prost-build::configure()`
    // returns the prost-based Builder we used pre-0.14.
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .file_descriptor_set_path(&descriptor_path)
        .compile_protos(
            &["proto/nvidia/omniverse/live/v1alpha/live_service.proto"],
            &["proto"],
        )
        .expect("failed to compile live service protobuf definitions");

    println!("cargo:rerun-if-changed=proto/nvidia/omniverse/live/v1alpha/live_service.proto");

    // ── External service protos (storage service + userinfo) ────────────────
    // Server code is also generated so that tests can stand up mock servers.
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                "proto/nvidia/omniverse/storage/fileobject/v1beta/fileobject_service.proto",
                "proto/nvidia/omniverse/userinfo/v1alpha/user_info_service.proto",
            ],
            &["proto"],
        )
        .expect("failed to compile external service protobuf definitions");

    println!("cargo:rerun-if-changed=proto/nvidia/omniverse/storage/fileobject/v1beta/fileobject_service.proto");
    println!(
        "cargo:rerun-if-changed=proto/nvidia/omniverse/storage/fileobject/v1beta/fileobject.proto"
    );
    println!(
        "cargo:rerun-if-changed=proto/nvidia/omniverse/userinfo/v1alpha/user_info_service.proto"
    );

    // Static linkage of libstdc++ on Linux (required because the
    // distroless/cc runtime image ships no libstdc++.so) is forced from
    // `oodf-ffi/build.rs` via `cpp_link_stdlib(None)` + an explicit
    // `cargo:rustc-link-lib=static=stdc++`. Three earlier locations failed
    // (here, `oodf-ffi/build.rs` via `cargo:rustc-link-arg`, and
    // `.cargo/config.toml` rustflags — all relying on gcc's
    // `-static-libstdc++` driver flag, which gcc 14 did not honor in this
    // build path). See SDD §8.2.
}
