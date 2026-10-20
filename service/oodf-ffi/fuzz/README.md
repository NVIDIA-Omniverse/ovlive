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

# `oodf-ffi` fuzz harness

`cargo-fuzz` targets that complement the upstream FlatBuffers verifier
fuzzing by exercising our Rust FFI + conversion layer directly. Three
targets, in order of structural awareness:

1. `fuzz_parse_delta` — feeds random bytes into `parse_delta` and asserts
   the parser never panics. Catches FFI / conversion-layer regressions on
   wholly-malformed input.
2. `fuzz_mutate_built` — same shape as `fuzz_parse_delta`
   but the recommended seed corpus is `tests/fixtures/oodf/*.oodf`, so the
   libfuzzer mutator produces inputs that share the layout of legitimate
   OODF traffic and stress the post-verifier accessor path.
3. `fuzz_round_trip` — structure-aware: generates an
   arbitrary `OwnedDelta` from the libfuzzer byte stream and asserts
   `parse(build(x)) == x`. Catches build → parse asymmetries and schema-
   drift regressions.

## Prerequisites

- Rust nightly (libfuzzer requires `#![no_main]` and nightly intrinsics).
- `cargo install cargo-fuzz`.
- Same system libraries needed to build `oodf-ffi` (a C++ toolchain plus
  whatever `flatc` the top-level `build.rs` calls out to).

## Run locally

From the workspace root:

```sh
cd oodf-ffi

# Random-byte parse fuzzer
cargo +nightly fuzz run fuzz_parse_delta -- -max_total_time=60

# Structure-aware round-trip
cargo +nightly fuzz run fuzz_round_trip -- -max_total_time=60

# Mutate-built-bytes — seed from fixtures first
mkdir -p corpus/fuzz_mutate_built
cp ../tests/fixtures/oodf/*.oodf corpus/fuzz_mutate_built/
cargo +nightly fuzz run fuzz_mutate_built -- -max_total_time=60
```

For extended soak runs, bump the time budget (e.g. `-max_total_time=14400`
for 4 hours).

libFuzzer writes each target's evolving corpus to `corpus/<target>/` and any
crash reproducers to `artifacts/<target>/` (both gitignored). A
non-zero exit status from `cargo fuzz run` indicates a reproducer was found —
treat it as a high-priority bug.

## Layout

- `Cargo.toml` — independent package (excluded from the workspace), standard
  cargo-fuzz manifest with `libfuzzer-sys = "0.4"` and `arbitrary` for the
  round-trip target. Enables the `oodf-ffi` `internal-bridge` feature so the
  fuzz crate can reach bridge types directly when needed.
- `fuzz_targets/fuzz_parse_delta.rs` — random-byte parse fuzzer.
- `fuzz_targets/fuzz_mutate_built.rs` — mutate-built-bytes fuzzer.
- `fuzz_targets/fuzz_round_trip.rs` — structure-aware round-trip fuzzer.
- `.gitignore` — `target/`, `corpus/`, `artifacts/`, `coverage/`.

## OMPF coverage

This harness targets `oodf_ffi::parse_delta` only. OMPF container parsing
lives in the `live-edit` crate at `src/domain/ompf.rs`, not in `oodf-ffi`,
so a `fuzz_parse_ompf` target here would require adding `live-edit` (with
its tonic + tokio + fred + axum dependency surface) to the fuzz crate.
The benefit-to-build-cost is not worth it for a single 200-line parser.

The equivalent OMPF coverage is spread across:

- **Random-byte parse-never-panic**: proptest
  `proptest_random_bytes_never_panic` in `tests/proptests.rs` runs 512
  cases per invocation (overridable via `PROPTEST_CASES`).
- **Adversarial header corpus**: `tests/misc/ompf_adversarial_corpus.rs` pins
  two attack shapes (`num_parts = u32::MAX` OOM header,
  `offsets[0]` inside the offset table) and a grab-bag of malformed
  byte strings as always-run integration tests.

If OMPF moves into `oodf-ffi` (e.g. to share header types with a streaming
parser), add a `fuzz_parse_ompf` target here that mirrors
`fuzz_parse_delta` — seed the corpus from
`tests/misc/ompf_adversarial_corpus.rs`'s byte constructors.

## CI integration

CI runs `fuzz_parse_delta` for 60 seconds, from `oodf-ffi/`, as the
`fuzz:parse_delta` job on merge-request, branch and manual-tag pipelines:

```sh
cargo +nightly fuzz run fuzz_parse_delta -- -max_total_time=60
```

The run starts from an empty corpus. A crash fails the job, which keeps
`oodf-ffi/fuzz/artifacts/` for a week so the reproducer can be re-run
locally. `fuzz_mutate_built` and `fuzz_round_trip` run only locally.
