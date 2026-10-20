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
use std::path::{Path, PathBuf};

// README pins flatc 24.3.25 to match the vendored schema. In practice
// FlatBuffers has been wire-format-compatible across 23/24/25.x for the
// `--cpp` codegen we use (the generated headers compile identically for our
// schema), so we accept this range and hard-fail outside it. The pinned
// major — used to emit a cargo warning when the local flatc does not match
// exactly — lives here rather than inline so the README's pin and build
// guard stay visibly coupled.
const FLATC_PINNED_MAJOR: u32 = 24;
const FLATC_MIN_MAJOR: u32 = 23;
const FLATC_MAX_MAJOR: u32 = 25;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    // ── FlatBuffers / flatc (--cpp) ─────────────────────────────────────────
    let flatc = which::which("flatc").unwrap_or_else(|_| {
        panic!(
            "\n\
            ============================================================\n\
            ERROR: `flatc` (FlatBuffers compiler) is not installed or\n\
            not found in PATH.\n\
            \n\
            Install it via your package manager, e.g.:\n\
              macOS:  brew install flatbuffers\n\
              Ubuntu: apt-get install flatbuffers-compiler\n\
            or download a release from:\n\
              https://github.com/google/flatbuffers/releases\n\
            ============================================================\n"
        )
    });

    check_flatc_major_version(&flatc);

    let fbs_schema = "schema/OmniObjectDelta.fbs";
    let flatc_out_dir = out_dir.to_str().expect("OUT_DIR is not valid UTF-8");

    let status = std::process::Command::new(&flatc)
        .args(["--cpp", "--no-warnings", "-o", flatc_out_dir, fbs_schema])
        .status()
        .unwrap_or_else(|e| panic!("failed to run flatc: {e}"));

    if !status.success() {
        panic!("flatc exited with non-zero status: {status}");
    }

    // Resolve the FlatBuffers C++ include path so that the generated header
    // can `#include "flatbuffers/flatbuffers.h"`. flatc is installed alongside
    // its headers under `<prefix>/include/flatbuffers/`, so derive the prefix
    // from the resolved flatc path: `<prefix>/bin/flatc` → `<prefix>/include`.
    let flatbuffers_include = flatc
        .parent()
        .and_then(|bin| bin.parent())
        .map(|prefix| prefix.join("include"))
        .unwrap_or_else(|| {
            panic!("could not derive flatbuffers include path from flatc: {flatc:?}")
        });

    // ── cxx-build: compile generated bridge + stub C++ ──────────────────────
    let mut cxx_builder = cxx_build::bridge("src/bridge.rs");
    cxx_builder
        .file("cpp/oodf.cpp")
        .include("include")
        .include(&out_dir)
        .include(&flatbuffers_include)
        .flag_if_supported("-std=c++17");

    // On Linux, suppress cc-rs's default `cargo:rustc-link-lib=stdc++` emission
    // (which produces a dynamic `-l stdc++` to the linker, and surfaces in the
    // final binary as a `libstdc++.so` DT_NEEDED entry). The distroless/cc
    // runtime image ships no libstdc++.so, so any dynamic dependency would
    // crash on startup; the Dockerfile `ldd` guard fails the build if one
    // survives. Two prior attempts to fix this via `-static-libstdc++` —
    // emitted from `oodf-ffi/build.rs`, then `build.rs` at the repo root,
    // then `.cargo/config.toml` `rustflags` — all failed: the gcc driver flag
    // landed in a position where `-lstdc++` had already been resolved to the
    // shared object before the static-libstdc++ directive took effect, or gcc
    // 14 silently dropped the flag.
    //
    // Bulletproof fix: disable the auto-emit, then explicitly emit
    // `cargo:rustc-link-lib=static=stdc++` below. rustc's `static=` kind
    // translates to `-Wl,-Bstatic -l stdc++ -Wl,-Bdynamic` brackets around
    // the library, forcing ld to pick `libstdc++.a` regardless of gcc's
    // driver-level resolution.
    #[cfg(target_os = "linux")]
    {
        cxx_builder.cpp_link_stdlib(None::<&str>);
    }

    cxx_builder.compile("oodf-ffi");

    // Force static linkage of libstdc++ on Linux (paired with the
    // `cpp_link_stdlib(None)` suppression above). `cargo:rustc-link-lib`
    // propagates from this library crate's build.rs to every downstream
    // binary that depends on `oodf-ffi`, including the `live-edit` release
    // binary that gets baked into the distroless/cc runtime image.
    //
    // rustc's default native-library search path does NOT include gcc's
    // per-version directory (typically `/usr/lib/gcc/x86_64-linux-gnu/14/`)
    // where `libstdc++.a` from `libstdc++-N-dev` lives — so we ask g++
    // directly for the absolute path and emit a `cargo:rustc-link-search`
    // before the link-lib request.
    #[cfg(target_os = "linux")]
    {
        emit_libstdcxx_search_path();
        println!("cargo:rustc-link-lib=static=stdc++");
        // GNU ld scans our bundled runtime before cxx's separate archive,
        // whose iostream initializer can introduce late unresolved symbols.
        // Rescan the static runtime AFTER the Rust archives for examples.
        // libgcc supplies libstdc++'s outlined atomic helpers on aarch64.
        // Link arguments do not propagate to the service or alter staticlib
        // contents; keep the bundled runtime above for those consumers.
        // The service Docker context excludes examples entirely; Cargo
        // rejects an example-only directive when no example target exists.
        println!("cargo:rerun-if-changed=examples");
        if Path::new("examples/decode_ompf.rs").is_file() {
            println!("cargo:rustc-link-arg-examples=-Wl,-Bstatic,-lstdc++,-lgcc,-Bdynamic");
        }
    }

    // ── Compile-only header self-containment check ─────────────────────────
    // Compile `cpp/tests/header_compiles.cpp` in isolation against just
    // `include/oodf.h` + the cxx-generated headers (no flatc-generated header,
    // no other TUs). Any missing include in the public header surfaces as a
    // compile error here rather than downstream in Epic 3's bridge code. The
    // resulting archive is not linked into any Rust artifact — only its
    // successful compilation matters.
    //
    // The cxx bridge headers live under `$OUT_DIR/cxxbridge/include/` after
    // `cxx_build::bridge(...)` has run.
    let cxx_include = out_dir.join("cxxbridge").join("include");
    cc::Build::new()
        .cpp(true)
        .file("cpp/tests/header_compiles.cpp")
        .include("include")
        .include(&cxx_include)
        .flag_if_supported("-std=c++17")
        .warnings(true)
        // Do not emit `cargo:rustc-link-lib` for this archive. The header-
        // self-containment check only cares that `include/oodf.h` compiles
        // against just the cxx-generated headers; the resulting `.a` is a
        // by-product, not a link input. Letting cargo auto-link it would
        // drag an empty archive into every downstream binary that depends
        // on oodf-ffi.
        .cargo_metadata(false)
        .compile("oodf-ffi-header-compiles");

    // ── Standalone C++ test binary (Bee b.duy Task 2.4) ────────────────────
    // The actual compile + link of the standalone C++ test binary is deferred
    // to the Rust wrapper at oodf-ffi/tests/cpp_tests.rs. build.rs cannot do
    // it here: the test binary links the `liboodf_ffi.a` staticlib that cargo
    // only emits AFTER build.rs finishes (chicken-and-egg). Instead, we
    // forward every path the wrapper needs via cargo:rustc-env so the test
    // can run the compiler itself with all knowledge build.rs already has.
    let cxx_runtime_include = cxx_runtime_include_dir();
    println!(
        "cargo:rustc-env=OODF_FFI_CPP_TEST_BUILD_DIR={}",
        out_dir.join("oodf_cpp_tests").display()
    );
    println!(
        "cargo:rustc-env=OODF_FFI_FLATC_OUT_DIR={}",
        out_dir.display()
    );
    println!(
        "cargo:rustc-env=OODF_FFI_FLATBUFFERS_INCLUDE={}",
        flatbuffers_include.display()
    );
    println!(
        "cargo:rustc-env=OODF_FFI_CXXBRIDGE_INCLUDE={}",
        cxx_include.display()
    );
    println!(
        "cargo:rustc-env=OODF_FFI_CXX_RUNTIME_INCLUDE={}",
        cxx_runtime_include.display()
    );
    println!(
        "cargo:rustc-env=OODF_FFI_MANIFEST_DIR={}",
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set")
    );
    // Forward the host target triple so the Rust-side test can match cargo's
    // profile-specific target directory where the staticlib gets emitted.
    println!(
        "cargo:rustc-env=OODF_FFI_TARGET={}",
        env::var("TARGET").unwrap_or_default()
    );

    println!("cargo:rerun-if-changed=schema/OmniObjectDelta.fbs");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=cpp/oodf.cpp");
    println!("cargo:rerun-if-changed=cpp/tests/header_compiles.cpp");
    println!("cargo:rerun-if-changed=cpp/tests/test_main.cpp");
    println!("cargo:rerun-if-changed=cpp/tests/fixtures.cpp");
    println!("cargo:rerun-if-changed=cpp/tests/fixtures.h");
    println!("cargo:rerun-if-changed=include/oodf.h");
    println!("cargo:rerun-if-changed=build.rs");
}

// Refuse to build against a flatc whose major version is outside the
// compatibility window declared at the top of this file. Emit a cargo
// warning when the version does not exactly match the README-pinned value,
// so contributors notice drift without being blocked on every minor release.
fn check_flatc_major_version(flatc: &Path) {
    let output = std::process::Command::new(flatc)
        .arg("--version")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke `{} --version`: {e}", flatc.display()));

    if !output.status.success() {
        panic!(
            "`{} --version` exited with status {}\n\
             --- stderr ---\n{}",
            flatc.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Canonical format is `flatc version 24.3.25\n`. The leading tokens are
    // fixed, so splitting on whitespace and pulling the first dotted-number
    // run is robust against newline/whitespace drift.
    let version_token = stdout
        .split_whitespace()
        .find(|tok| {
            tok.split('.')
                .next()
                .is_some_and(|head| head.chars().all(|c| c.is_ascii_digit()))
                && tok.contains('.')
        })
        .unwrap_or_else(|| {
            panic!(
                "`flatc --version` output did not contain a recognizable \
                 `<major>.<minor>.<patch>` token: {stdout:?}"
            )
        });

    let major = version_token
        .split('.')
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or_else(|| {
            panic!("could not parse major version from flatc token {version_token:?}")
        });

    if !(FLATC_MIN_MAJOR..=FLATC_MAX_MAJOR).contains(&major) {
        panic!(
            "\n\
            ============================================================\n\
            ERROR: flatc major version {major} is outside the supported\n\
            range {FLATC_MIN_MAJOR}..={FLATC_MAX_MAJOR}.\n\
            \n\
            The vendored OODF schema was pinned against flatc {FLATC_PINNED_MAJOR}.x\n\
            (README.md, oodf-ffi/schema/SOURCE.md). Older or newer majors\n\
            may emit incompatible C++ codegen.\n\
            \n\
            Install a compatible release, e.g.:\n\
              macOS:  brew install flatbuffers@{FLATC_PINNED_MAJOR}\n\
              Ubuntu: apt-get install flatbuffers-compiler={FLATC_PINNED_MAJOR}.*\n\
            or download from:\n\
              https://github.com/google/flatbuffers/releases/tag/v{FLATC_PINNED_MAJOR}.3.25\n\
            \n\
            Observed: {version_token}\n\
            Path:     {}\n\
            ============================================================\n",
            flatc.display()
        );
    }

    if major != FLATC_PINNED_MAJOR {
        println!(
            "cargo:warning=oodf-ffi: local flatc is {version_token} but README pins \
             {FLATC_PINNED_MAJOR}.3.25. Accepted (within range \
             {FLATC_MIN_MAJOR}..={FLATC_MAX_MAJOR}), but consider installing the pinned \
             release if you see codegen diffs."
        );
    }
}

// Ask `g++ --print-file-name=libstdc++.a` for the absolute path of the
// static C++ stdlib archive, then emit `cargo:rustc-link-search=native=DIR`
// pointing at its parent directory. Required because rustc's default native-
// library search path does not include gcc's per-version directory where
// `libstdc++.a` is installed (typically `/usr/lib/gcc/x86_64-linux-gnu/14/`).
// Hard-fails if g++ cannot locate the archive — that means `libstdc++-N-dev`
// isn't installed and the static link cannot succeed.
#[cfg(target_os = "linux")]
fn emit_libstdcxx_search_path() {
    let output = std::process::Command::new("g++")
        .arg("--print-file-name=libstdc++.a")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke `g++ --print-file-name=libstdc++.a`: {e}"));
    if !output.status.success() {
        panic!(
            "`g++ --print-file-name=libstdc++.a` exited with status {}\n\
             --- stderr ---\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let raw = String::from_utf8(output.stdout)
        .expect("g++ --print-file-name=libstdc++.a stdout is not valid UTF-8");
    let path = Path::new(raw.trim());
    if !path.is_absolute() || !path.exists() {
        panic!(
            "could not locate libstdc++.a (g++ returned {raw:?}). Install \
             `libstdc++-N-dev` so the static archive is available alongside \
             the cxx C++ build."
        );
    }
    let dir = path.parent().unwrap_or_else(|| {
        panic!(
            "libstdc++.a path has no parent directory: {}",
            path.display()
        )
    });
    println!("cargo:rustc-link-search=native={}", dir.display());
}

// Derive the cxx crate's `include/` directory (home of `cxx.h`) from the
// `DEP_CXXBRIDGE1_HEADER` env var that cxx's build script exports. `cxx` is a
// direct dependency of `oodf-ffi` so this env var is always set for us.
fn cxx_runtime_include_dir() -> PathBuf {
    let cxx_header_env = env::var("DEP_CXXBRIDGE1_HEADER").unwrap_or_else(|_| {
        panic!(
            "DEP_CXXBRIDGE1_HEADER not set — the cxx crate is expected as a \
             direct dependency of oodf-ffi so its build script exports the \
             runtime header path"
        )
    });
    let cxx_header = PathBuf::from(cxx_header_env);
    cxx_header
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| {
            panic!(
                "cannot derive cxx runtime include dir from DEP_CXXBRIDGE1_HEADER={}",
                cxx_header.display()
            )
        })
}
