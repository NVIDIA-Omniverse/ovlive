<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# oodf-ffi

Foreign-Function-Interface library that gives the `live-edit` Rust service access to the **Omniverse Object Delta Format (OODF)** wire format via a small C++ library wrapped in a safe Rust API.

- **OODF** = **O**mniverse **O**bject **D**elta **F**ormat. The FlatBuffers schema that encodes every delta on the wire. Every valid OODF buffer carries the 4-byte file identifier `"OODF"` at bytes 4-7.
- **FFI** = Foreign Function Interface. Here: Rust ↔ C++ via the [`cxx`](https://cxx.rs) crate.

## Why this crate exists

The authoritative OODF schema uses a native FlatBuffers vector-of-unions for `structCommands` (see `schema/OmniObjectDelta.fbs`). **Rust's `flatc` code generator does not support vectors of unions** — confirmed empirically with `flatc 25.12.19` + `flatbuffers = "24"`, with and without `--gen-object-api`. **C++ `flatc` does.**

So this crate:

1. Ships the authoritative schema (`schema/OmniObjectDelta.fbs`).
2. Compiles it with `flatc --cpp` at build time.
3. Implements `parse_delta` and `build_delta` in C++ against the generated accessors/builders.
4. Exposes those two functions to Rust through a `cxx` bridge that uses plain owned types.

Rust code outside this crate never touches FlatBuffers. It handles idiomatic owned types (`OwnedDelta`, `StructCommand`, `SetField`, `TimeSample`, `OodfError`) and calls `oodf_ffi::parse_delta` / `oodf_ffi::build_delta`.

## Public Rust API

```rust
pub fn parse_delta(bytes: &[u8]) -> Result<OwnedDelta, OodfError>;
pub fn build_delta(delta: &OwnedDelta) -> Result<Vec<u8>, OodfError>;
pub const MAX_DELTA_TABLES: usize = 100_000; // one delta: the root + at most 99,999 entries
```

`parse_delta` never panics on any input — all malformed buffers come back as a specific `OodfError` variant. `build_delta` is fallible: an unknown `CommandTag` or `std::bad_alloc` comes back as an error rather than a `std::abort()`, which would take down every concurrent in-flight RPC sharing the process. The Rust idiomatic types make every well-formed `OwnedDelta` representable, so callers building deltas from owned Rust values will not normally see an error; the `Result` exists so a buggy bridge caller (smuggling a raw out-of-range `repr: u8`) or a real allocator exhaustion surfaces as data rather than a process abort. The reachable error variants on the build path are `OodfError::BuildUnknownCommandTag(String)`, `OodfError::AllocationFailed`, and `OodfError::InternalError(String)`.

The cxx `bridge` and `convert` modules are `pub(crate)` by default — downstream consumers cannot import `oodf_ffi::bridge::ffi` or any cxx shared struct and must go through the idiomatic types in `crate::types` plus the public wrappers above. The opt-in `internal-bridge` cargo feature re-exposes both modules as `#[doc(hidden)] pub` so the crate's own integration tests under `tests/` (which need to construct bridge values to exercise FFI edge cases) compile. The feature is NOT a default feature and is NOT enabled by `live-edit`'s `[dependencies] oodf-ffi = { path = ".." }` declaration; the workspace-wide `cargo clippy --workspace --all-targets` and `cargo test --workspace` invocations pass `--features oodf-ffi/internal-bridge` so the workspace test pass exercises both surfaces. Run the crate's own tests with `cargo test -p oodf-ffi --features internal-bridge`.

## Directory layout

```
oodf-ffi/
├── Cargo.toml          # crate-type = ["staticlib", "rlib"]; pins cxx + cxx-build
├── build.rs            # runs flatc --cpp; drives cxx-build; static-links libstdc++/libgcc on Linux
├── schema/             # VENDORED AUTHORITATIVE SCHEMA — see SOURCE.md for provenance + re-vendor steps
│   ├── OmniObjectDelta.fbs
│   └── SOURCE.md
├── src/                # Rust side: idiomatic types + #[cxx::bridge] + public API
│   └── lib.rs
├── include/            # C++ public header (plain-data types, no FlatBuffers leakage)
│   └── oodf.h
├── cpp/                # C++ implementation
│   ├── oodf.cpp        # parse_delta + build_delta
│   └── tests/          # standalone C++ test harness, driven by tests/cpp_tests.rs
├── tests/              # Rust-side integration tests
│   ├── cpp_tests.rs    # wraps the C++ test binary
│   ├── conversion.rs   # bridge ↔ idiomatic type round-trips
│   ├── malformed_input.rs  # negative-path coverage through parse_delta
│   ├── schema_integrity.rs # asserts the vendored .fbs hasn't drifted
│   └── …               # additional per-feature integration files
└── fuzz/               # cargo-fuzz harness — separate from `cargo test`
```

## Security invariant

**`parse_delta` runs the C++ FlatBuffers verifier before any accessor is touched on untrusted bytes, with one exception: the entry-count probe.** This is the single most important property of the crate — any PR that appears to call a generated accessor before `VerifyDeltaBuffer` returns true, or to add any other read before it, is a reviewer-reject. The probe (`probe_entry_counts`) runs just before the verifier and reads only the root offset, the root table's vtable, and the offset slots and length words of its entry vectors (`structCommands_type`, `structCommands`, `setFields`, `timeSamples`), each after the FlatBuffers library's own verify primitives have bounds-checked it; it never follows an element offset. A delta whose struct commands, set fields and time samples sum to more than `MAX_DELTA_TABLES - 1` (99,999) is refused `OodfError::TooManyTables`, carrying the three counts, before the verifier walks it; a buffer the probe cannot count falls through to the verifier unchanged. The invariant is stated in `cpp/oodf.cpp`'s header comment, and the order of the checks at the top of `parse_delta`.

The verifier is configured with tightened `Verifier::Options` (`max_depth = 8`, `max_tables = 100_000`) to reject deep-table-walk and oversized-vector DoS shapes that the library defaults would otherwise let through; its table cap is the backstop for a buffer the probe could not count. `max_tables` equals the Rust constant `MAX_DELTA_TABLES`, and the two change together. Verified-vector lengths are then capped at `kMaxCommands = 1_000_000` before any per-element accumulator runs — a verifier-accepted million-element vector inside an otherwise compact buffer would otherwise commit hundreds of MB on the consumer side per parse. For the three entry vectors that cap is unreachable while the probe and the table cap hold; a `ReorderChildren` children list over it is refused `OodfError::ChildrenListTooLong`, carrying its length. These caps are documented as the public contract that callers may rely on; bumping them, or adding a read before the verifier, is allowed but requires a security review (see the comment block above `kVerifierMaxDepth` / `kMaxCommands` in `cpp/oodf.cpp`).

**`build_delta` never calls `std::abort()`.** An unknown `CommandTag`, a `std::bad_alloc` or any other exception comes back through a typed `BuildResult` shared struct that the Rust wrapper decodes into `OodfError::BuildUnknownCommandTag` / `AllocationFailed` / `InternalError`, so it fails the one call rather than every concurrent in-flight RPC sharing the process. The in-function `try { ... } catch (...)` block in `cpp/oodf.cpp` is the sole boundary; no exception ever escapes into Rust.

## Build requirements

- Rust toolchain (see repo root `rust-toolchain.toml`).
- `flatc` 24.3.25 (or compatible) on `PATH`. On macOS: `brew install flatbuffers`. In CI and Docker it's built from source.
- A C++17 compiler (`g++`, `clang++`, or Apple Clang).
- On Linux release builds, the binary static-links `libstdc++` and `libgcc` regardless of what the distroless/cc runtime image ships, to avoid libstdc++ version skew against the base image; the Dockerfile `ldd` guard enforces it.

See [Prerequisites](../README.md#prerequisites) under Build from source in the top-level README for full dev-machine setup.

## Running tests

```sh
# All Rust-level tests. The `internal-bridge` feature opens up the
# `pub(crate)` cxx bridge module so the integration tests under `tests/`
# (which need to construct bridge values to exercise FFI edge cases) can
# compile. Production builds and `live-edit`'s dependency edge do NOT
# enable the feature.
cargo test -p oodf-ffi --features internal-bridge

# The C++-level standalone test suite (invoked via the Rust wrapper)
cargo test -p oodf-ffi --features internal-bridge --test cpp_tests

# Fuzz harness (nightly Rust required; not part of `cargo test`).
# See `fuzz/README.md` for the full set of targets, including the
# structure-aware `fuzz_round_trip` and seeded `fuzz_mutate_built`
# targets.
cargo +nightly fuzz run fuzz_parse_delta -- -max_total_time=60
cargo +nightly fuzz run fuzz_round_trip -- -max_total_time=60
cargo +nightly fuzz run fuzz_mutate_built -- -max_total_time=60
```

## Re-vendoring the schema

See `schema/SOURCE.md`. The copy differs from upstream only in its license notice; the recorded SHA256s of the whole file and of the schema body below the notice are enforced by the GitLab CI `schema:drift-check` job and by the Rust-side `tests/schema_integrity.rs` test.

## Why FFI, and not an upstream PR to Rust flatbuffers?

The obvious alternative to this crate is to contribute vector-of-unions support to the [`flatbuffers` Rust code generator](https://github.com/google/flatbuffers) upstream. If that lands, everything in this crate becomes unnecessary — the service could use Rust flatc directly. That's worth doing at some point.

We are **not** pursuing it in the near term. Reasons:

- **Timeline uncertainty.** Upstream PR cycles for cross-language FlatBuffers features can run weeks to months. We needed drop-in wire compatibility with the legacy C++ `live-edit` service now, not "sometime after a PR lands on crates.io."
- **Version pinning pain even after merge.** Consumers (us, CI, Docker) would need the specific `flatbuffers` crate version that includes the feature — plus the matching `flatc` binary in every environment including CI and the Docker builder. That's still toolchain coordination work.
- **Review scope creep risk.** FlatBuffers is a cross-language project and upstream reviewers (reasonably) ask for broader scope: language-feature parity, test coverage across generators, etc. A straightforward "add vectors of unions to Rust gen" PR can balloon.
- **OODF schema is stable.** The authoritative `.fbs` has been unchanged since 2019. This crate isn't carrying ongoing maintenance burden — once it works, it works.
- **Reversible.** If and when the upstream PR merges, `live-edit` can delete this crate entirely. Nothing about the FFI approach locks us in.

**What we'd want if someone does pursue the upstream PR later:** file a bug ticket referencing this rationale, link to the PR, and when it merges on `crates.io`, replace `oodf-ffi`'s public API with a pure-Rust implementation behind the same `oodf_ffi::parse_delta` / `oodf_ffi::build_delta` surface. Callers upstream of this crate wouldn't need to change.
