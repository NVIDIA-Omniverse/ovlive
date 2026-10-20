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

//! Compile and run the standalone C++ test binary.
//!
//! This wrapper exists because build.rs cannot produce the test executable:
//! the executable links the `liboodf_ffi.a` staticlib that cargo only emits
//! *after* build.rs completes. So we defer the compile + link to this Rust
//! test — at `cargo test` time the staticlib is already on disk, and we
//! invoke the C++ compiler from here with paths forwarded by build.rs via
//! `cargo:rustc-env`.
//!
//! # Required prerequisite
//!
//! `cargo test -p oodf-ffi --test cpp_tests` expects `liboodf_ffi.a` (or
//! `oodf_ffi.lib` on MSVC) to already exist in the active profile directory.
//! Cargo does NOT build the staticlib as part of an integration-test run —
//! integration-test targets only depend on the `rlib` crate-type. Before
//! running this test, run one of the following to materialize the archive:
//!
//! ```text
//! cargo build -p oodf-ffi --lib
//! cargo build --workspace
//! cargo test --workspace          # builds the staticlib indirectly via workspace members
//! ```
//!
//! CI pipelines and the pre-commit `cargo clippy --workspace --all-targets`
//! invocation already do this, so the prerequisite is met there. The earlier
//! version of this test tried to handle the missing-artifact case by
//! shelling out to `cargo build -p oodf-ffi --lib` from inside the test,
//! which risked target-lock contention and CI deadlocks under nested cargo
//! invocations. Failing loudly with instructions is safer — the fix is a
//! single extra command on local dev machines. `CLAUDE.md` documents the
//! same prerequisite under "Build & Test Commands".
//!
//! Every actual test case lives in `oodf-ffi/cpp/tests/` (test_main.cpp and
//! sibling TUs added by subtasks 2.4.B/C). This file only orchestrates the
//! build and asserts a clean exit.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

// Emitted by build.rs (Task 2.4 scaffolding — see `oodf-ffi/build.rs`).
const CPP_TEST_BUILD_DIR: &str = env!("OODF_FFI_CPP_TEST_BUILD_DIR");
const FLATC_OUT_DIR: &str = env!("OODF_FFI_FLATC_OUT_DIR");
const FLATBUFFERS_INCLUDE: &str = env!("OODF_FFI_FLATBUFFERS_INCLUDE");
const CXXBRIDGE_INCLUDE: &str = env!("OODF_FFI_CXXBRIDGE_INCLUDE");
const CXX_RUNTIME_INCLUDE: &str = env!("OODF_FFI_CXX_RUNTIME_INCLUDE");
const MANIFEST_DIR: &str = env!("OODF_FFI_MANIFEST_DIR");

// 60s wall-clock cap on the child. The suite is pure CPU work with no I/O, so
// under any reasonable CI this completes in well under a second; the cap only
// exists to turn a future infinite loop into a clear Rust-side test failure
// rather than a hung cargo run.
const CHILD_TIMEOUT: Duration = Duration::from_secs(60);

#[test]
fn cpp_test_binary_runs_and_exits_zero() {
    let bin_path = compile_cpp_test_binary();
    let output = run_with_timeout(&bin_path, CHILD_TIMEOUT);

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "\noodf-ffi C++ test binary exited non-zero ({})\n\
             binary: {}\n\
             --- stdout ---\n{}\n\
             --- stderr ---\n{}\n",
            output.status,
            bin_path.display(),
            stdout,
            stderr
        );
    }
}

// Compile + link the C++ test binary against the `liboodf_ffi.a` staticlib
// that cargo has already emitted for this test run. The binary is cached at
// `$OUT_DIR/oodf_cpp_tests/oodf_cpp_tests` — we rebuild unconditionally on
// every test run because the cost is small (~1s) and staleness detection here
// would duplicate cargo's job.
fn compile_cpp_test_binary() -> PathBuf {
    let build_dir = Path::new(CPP_TEST_BUILD_DIR);
    std::fs::create_dir_all(build_dir).expect("failed to create C++ test build dir");

    let exe_suffix = if cfg!(windows) { ".exe" } else { "" };
    let bin_path = build_dir.join(format!("oodf_cpp_tests{exe_suffix}"));

    let staticlib = ensure_staticlib();
    let manifest = Path::new(MANIFEST_DIR);
    let sources = collect_test_sources(&manifest.join("cpp").join("tests"));

    let compiler = pick_compiler();

    let mut cmd = Command::new(&compiler);
    if cfg!(target_env = "msvc") {
        cmd.arg("/std:c++17");
        cmd.arg("/EHsc");
        for inc in include_dirs() {
            cmd.arg(format!("/I{}", inc.display()));
        }
        for src in &sources {
            cmd.arg(src);
        }
        cmd.arg(format!("/Fe{}", bin_path.display()));
        cmd.arg("/link");
        cmd.arg(&staticlib);
    } else {
        cmd.arg("-std=c++17");
        for inc in include_dirs() {
            cmd.arg("-I").arg(inc);
        }
        for src in &sources {
            cmd.arg(src);
        }
        // The staticlib carries oodf.cpp, the cxx runtime, and — crucially —
        // the `_cxxbridge1$*` Rust-side symbols that `rust::String` / `rust::Vec`
        // dispatch to. Pass its absolute path as a positional input.
        cmd.arg(&staticlib);
        cmd.arg("-o").arg(&bin_path);
        // On macOS link against frameworks the Rust staticlib needs.
        #[cfg(target_os = "macos")]
        {
            cmd.arg("-framework").arg("CoreFoundation");
            cmd.arg("-framework").arg("Security");
        }
        // On linux, -lpthread / -ldl / -lm are needed for the Rust std runtime
        // that cargo normally supplies via its staticlib-linking logic.
        #[cfg(target_os = "linux")]
        {
            cmd.arg("-lpthread");
            cmd.arg("-ldl");
            cmd.arg("-lm");
            cmd.arg("-lrt");
        }
    }

    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke {compiler:?} to build C++ test binary: {e}"));
    if !output.status.success() {
        panic!(
            "\nC++ compile/link of oodf_cpp_tests failed ({})\n\
             compiler: {:?}\n\
             staticlib: {}\n\
             --- stdout ---\n{}\n\
             --- stderr ---\n{}\n",
            output.status,
            compiler,
            staticlib.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    bin_path
}

// Collect every `*.cpp` in `cpp/tests/` except the compile-only self-
// containment check (`header_compiles.cpp`). Keeping this glob-driven means
// subtasks 2.4.B/C can drop new TUs next to `test_main.cpp` without touching
// this file or build.rs.
fn collect_test_sources(dir: &Path) -> Vec<PathBuf> {
    const SKIP: &[&str] = &["header_compiles.cpp"];
    let mut out = Vec::new();
    for entry in
        std::fs::read_dir(dir).unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()))
    {
        let entry =
            entry.unwrap_or_else(|e| panic!("dir-entry read failed under {}: {e}", dir.display()));
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("cpp") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if SKIP.contains(&name) {
            continue;
        }
        out.push(path);
    }
    // Sort for deterministic link order (and deterministic test-case registry
    // order; the harness executes tests in the order they register).
    out.sort();
    assert!(
        !out.is_empty(),
        "no C++ test sources found under {}",
        dir.display()
    );
    out
}

fn include_dirs() -> Vec<PathBuf> {
    vec![
        Path::new(MANIFEST_DIR).join("include"),
        Path::new(MANIFEST_DIR).join("cpp").join("tests"),
        PathBuf::from(CXXBRIDGE_INCLUDE),
        PathBuf::from(CXX_RUNTIME_INCLUDE),
        PathBuf::from(FLATBUFFERS_INCLUDE),
        PathBuf::from(FLATC_OUT_DIR),
    ]
}

// Pick a C++ compiler for the link step. Respect `CXX` if set (matches what
// cc::Build does in build.rs); otherwise fall back to `c++` on unix and `cl`
// on msvc. Production builds go through cc::Build, so this path is only used
// for the test binary.
fn pick_compiler() -> PathBuf {
    if let Ok(cxx) = std::env::var("CXX") {
        if !cxx.is_empty() {
            return PathBuf::from(cxx);
        }
    }
    if cfg!(target_env = "msvc") {
        PathBuf::from("cl")
    } else {
        PathBuf::from("c++")
    }
}

// Require `liboodf_ffi.a` to already exist in the active profile directory.
// See the module-level doc-comment for the rationale — in short: cargo does
// not build staticlibs for integration-test targets, and shelling out to
// `cargo build` from inside a test risked target-lock contention under CI's
// nested cargo invocations. Fail loudly with actionable instructions.
fn ensure_staticlib() -> PathBuf {
    if let Some(p) = locate_staticlib_path() {
        return p;
    }
    panic!(
        "\n\
        ============================================================\n\
        ERROR: {file_name} is not present in the active cargo profile\n\
        directory. This integration test links it to build the C++ test\n\
        binary.\n\
        \n\
        Run one of the following before re-running this test:\n\
          cargo build -p oodf-ffi --lib\n\
          cargo build --workspace\n\
        \n\
        (CI pipelines run `cargo build -p oodf-ffi --lib` as a dedicated\n\
        step before `cargo test --workspace`. `cargo clippy --all-targets`\n\
        does NOT emit the staticlib — it is check-semantics, no link.)\n\
        ============================================================\n",
        file_name = staticlib_file_name()
    );
}

// Scan the active profile directory plus triple-specific subdirectories for
// an existing staticlib. Returns `None` if none found.
fn locate_staticlib_path() -> Option<PathBuf> {
    let current_exe = std::env::current_exe().ok()?;
    let profile_dir = current_exe.parent()?.parent()?;
    let file_name = staticlib_file_name();
    let primary = profile_dir.join(&file_name);
    if primary.is_file() {
        return Some(primary);
    }
    let target_dir = profile_dir.parent()?;
    let profile_name = profile_dir.file_name()?;
    for entry in std::fs::read_dir(target_dir).ok()?.flatten() {
        let candidate = entry.path().join(profile_name).join(&file_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn staticlib_file_name() -> String {
    if cfg!(target_env = "msvc") {
        "oodf_ffi.lib".to_string()
    } else {
        "liboodf_ffi.a".to_string()
    }
}

struct CapturedOutput {
    status: std::process::ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn run_with_timeout(bin: &Path, timeout: Duration) -> CapturedOutput {
    // `std::process::Child::wait` has no portable timeout, so we run the
    // child on a worker thread and recv_timeout the result channel.
    let bin = bin.to_path_buf();
    let (tx, rx) = mpsc::channel();
    let handle = thread::spawn(move || {
        let result = Command::new(&bin).output();
        let _ = tx.send(result);
    });

    match rx.recv_timeout(timeout) {
        Ok(Ok(output)) => {
            let _ = handle.join();
            CapturedOutput {
                status: output.status,
                stdout: output.stdout,
                stderr: output.stderr,
            }
        }
        Ok(Err(e)) => {
            let _ = handle.join();
            panic!("failed to spawn C++ test binary: {e}");
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!(
                "oodf-ffi C++ test binary did not finish within {:?}; \
                 suspect an infinite loop in a test case",
                timeout
            );
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            let _ = handle.join();
            panic!("C++ test binary worker thread died without producing a result");
        }
    }
}
