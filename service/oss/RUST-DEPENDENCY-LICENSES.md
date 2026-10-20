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

# Rust dependency licenses

This table is a snapshot of the root `Cargo.lock` whose SHA-256 is `ec9fb4c45ebe73fc7990ad3002c4c7e7dcb9aaef7201e713213a3f21f17559a6`.
It lists all 395 packages that lockfile records, one row per name and version, sorted by name and then version.
The conformance suite's native codec has its own lockfile, `conformance_tests/native-Cargo.lock`; every package in it is also in the root lockfile at the same version, so this table covers it.
The fuzz harness's own lockfile, `oodf-ffi/fuzz/Cargo.lock`, is not covered: only `cargo-fuzz` builds the harness, and no release artifact ships it.
It was generated once, from `cargo metadata --locked --format-version 1`, and is not kept current: it is accurate as of the `Cargo.lock` whose SHA-256 it records, and after a dependency change, `Cargo.lock` and `ATTRIBUTIONS.md` are the current record.
The inventory of what each release ships and how it is licensed is [SOURCE-INVENTORY.md](SOURCE-INVENTORY.md).

## Columns

- **License** is the expression the package's own `Cargo.toml` declares.
  The vendored `fred` row instead gives the split recorded in `vendor/fred-10.1.0.SOURCE.md`: upstream files and NVIDIA's modifications to them are MIT, and NVIDIA's added files are Apache-2.0.
  `ATTRIBUTIONS.md` normalises two declarations: `smartstring` (`MPL-2.0+`) is attributed as `MPL-2.0`, and `tikv-jemalloc-sys` (`MIT/Apache-2.0`) as `(MIT OR Apache-2.0) AND BSD-2-Clause`, because it compiles jemalloc's BSD-2-Clause C sources into the binary.
- **Source** is `crates.io`, a workspace member's path, or the vendored tree the root `Cargo.toml` patches in with `[patch.crates-io]`.
- **Scope** comes from the dependency kinds `cargo metadata` reports, walked from the four workspace members (`live-edit`, `oodf-ffi`, `live-edit-capabilities-proto`, `live-edit-scaler`):
  - `runtime`: reachable through normal dependency edges only. On a platform where it builds (the **Linux** column), such a crate is compiled into the binaries the workspace crates produce.
    `runtime (proc macro)` marks a procedural-macro crate in that set: it runs inside the compiler, and none of its code is linked into a binary.
  - `build`: not `runtime`, and reachable once build-dependency edges are followed too. These run only while compiling: build scripts and what they use.
  - `dev-test`: reachable only through a dev-dependency edge. These build tests, benchmarks and examples, and are compiled into no release binary.
  - `not built`: recorded in `Cargo.lock` but outside every graph `cargo metadata` resolves. The only one is `cxxbridge-cmd`, a build dependency of `cxx` declared under the never-true `cfg(any())`.
  A package reached several ways takes the first of `runtime`, `build` and `dev-test` that any path gives it, so a package that both a runtime crate and a test use is `runtime`.
- **Linux** is `yes` when the package is in the dependency graph `cargo metadata --filter-platform` resolves for `x86_64-unknown-linux-gnu` or `aarch64-unknown-linux-gnu`.
  A `no` package is locked only for other platforms (such as Windows, macOS, WebAssembly or UEFI) and is not compiled into the Linux service binary.
  The service image's `/vendor.tar.gz` still carries its source, since it vendors the whole lockfile.

Scope totals: build 47, dev-test 38, not built 1, runtime 288, runtime (proc macro) 21.

## Packages

| Package | Version | License | Source | Scope | Linux |
|---|---|---|---|---|---|
| aho-corasick | 1.1.4 | Unlicense OR MIT | crates.io | runtime | yes |
| alloca | 0.4.0 | MIT | crates.io | dev-test | yes |
| android_system_properties | 0.1.5 | MIT/Apache-2.0 | crates.io | runtime | no |
| anes | 0.1.6 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| anstyle | 1.0.14 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| anyhow | 1.0.104 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| arc-swap | 1.9.2 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| assert-json-diff | 2.0.2 | MIT | crates.io | dev-test | yes |
| async-trait | 0.1.91 | MIT OR Apache-2.0 | crates.io | runtime (proc macro) | yes |
| atomic-waker | 1.1.2 | Apache-2.0 OR MIT | crates.io | runtime | yes |
| autocfg | 1.5.1 | Apache-2.0 OR MIT | crates.io | build | yes |
| aws-config | 1.10.1 | Apache-2.0 | crates.io | runtime | yes |
| aws-credential-types | 1.3.0 | Apache-2.0 | crates.io | runtime | yes |
| aws-runtime | 1.9.1 | Apache-2.0 | crates.io | runtime | yes |
| aws-sdk-sts | 1.110.0 | Apache-2.0 | crates.io | runtime | yes |
| aws-sigv4 | 1.5.1 | Apache-2.0 | crates.io | runtime | yes |
| aws-smithy-async | 1.3.0 | Apache-2.0 | crates.io | runtime | yes |
| aws-smithy-http | 0.64.0 | Apache-2.0 | crates.io | runtime | yes |
| aws-smithy-http-client | 1.2.0 | Apache-2.0 | crates.io | runtime | yes |
| aws-smithy-json | 0.63.0 | Apache-2.0 | crates.io | runtime | yes |
| aws-smithy-observability | 0.3.0 | Apache-2.0 | crates.io | runtime | yes |
| aws-smithy-query | 0.62.0 | Apache-2.0 | crates.io | runtime | yes |
| aws-smithy-runtime | 1.12.1 | Apache-2.0 | crates.io | runtime | yes |
| aws-smithy-runtime-api | 1.14.0 | Apache-2.0 | crates.io | runtime | yes |
| aws-smithy-runtime-api-macros | 1.1.0 | Apache-2.0 | crates.io | runtime (proc macro) | yes |
| aws-smithy-schema | 0.2.0 | Apache-2.0 | crates.io | runtime | yes |
| aws-smithy-types | 1.6.1 | Apache-2.0 | crates.io | runtime | yes |
| aws-smithy-xml | 0.62.0 | Apache-2.0 | crates.io | runtime | yes |
| aws-types | 1.5.0 | Apache-2.0 | crates.io | runtime | yes |
| axum | 0.7.9 | MIT | crates.io | runtime | yes |
| axum | 0.8.9 | MIT | crates.io | runtime | yes |
| axum-core | 0.4.5 | MIT | crates.io | runtime | yes |
| axum-core | 0.5.6 | MIT | crates.io | runtime | yes |
| base64 | 0.22.1 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| base64-simd | 0.8.0 | MIT | crates.io | runtime | yes |
| bit-set | 0.8.0 | Apache-2.0 OR MIT | crates.io | dev-test | yes |
| bit-vec | 0.8.0 | Apache-2.0 OR MIT | crates.io | dev-test | yes |
| bitflags | 2.13.1 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| block-buffer | 0.10.4 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| block-buffer | 0.12.1 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| bumpalo | 3.20.3 | MIT OR Apache-2.0 | crates.io | runtime | no |
| bytecount | 0.6.9 | Apache-2.0/MIT | crates.io | build | no |
| bytes | 1.12.1 | MIT | crates.io | runtime | yes |
| bytes-utils | 0.1.4 | Apache-2.0/MIT | crates.io | runtime | yes |
| camino | 1.2.5 | MIT OR Apache-2.0 | crates.io | build | no |
| cargo-platform | 0.1.9 | MIT OR Apache-2.0 | crates.io | build | no |
| cargo_metadata | 0.14.2 | MIT | crates.io | build | no |
| cast | 0.3.0 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| cc | 1.2.60 | MIT OR Apache-2.0 | crates.io | build | yes |
| cfg-if | 1.0.4 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| cfg_aliases | 0.2.2 | MIT | crates.io | build | yes |
| chacha20 | 0.10.2 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| chrono | 0.4.45 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| ciborium | 0.2.2 | Apache-2.0 | crates.io | dev-test | yes |
| ciborium-io | 0.2.2 | Apache-2.0 | crates.io | dev-test | yes |
| ciborium-ll | 0.2.2 | Apache-2.0 | crates.io | dev-test | yes |
| clap | 4.6.4 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| clap_builder | 4.6.2 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| clap_lex | 1.1.0 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| cmov | 0.5.4 | Apache-2.0 OR MIT | crates.io | runtime | yes |
| codespan-reporting | 0.13.1 | Apache-2.0 | crates.io | build | yes |
| const-oid | 0.10.2 | Apache-2.0 OR MIT | crates.io | runtime | yes |
| cookie-factory | 0.3.2 | MIT | crates.io | runtime | yes |
| core-foundation | 0.10.1 | MIT OR Apache-2.0 | crates.io | runtime | no |
| core-foundation-sys | 0.8.7 | MIT OR Apache-2.0 | crates.io | runtime | no |
| cpufeatures | 0.2.17 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| cpufeatures | 0.3.0 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| crc16 | 0.4.0 | MIT | crates.io | runtime | yes |
| criterion | 0.8.2 | Apache-2.0 OR MIT | crates.io | dev-test | yes |
| criterion-plot | 0.8.2 | Apache-2.0 OR MIT | crates.io | dev-test | yes |
| crossbeam-channel | 0.5.16 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| crossbeam-utils | 0.8.22 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| crunchy | 0.2.4 | MIT | crates.io | dev-test | no |
| crypto-common | 0.1.7 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| crypto-common | 0.2.2 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| ctutils | 0.4.2 | Apache-2.0 OR MIT | crates.io | runtime | yes |
| cxx | 1.0.195 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| cxx-build | 1.0.195 | MIT OR Apache-2.0 | crates.io | build | yes |
| cxxbridge-cmd | 1.0.195 | MIT OR Apache-2.0 | crates.io | not built | no |
| cxxbridge-flags | 1.0.195 | MIT OR Apache-2.0 | crates.io | build | yes |
| cxxbridge-macro | 1.0.195 | MIT OR Apache-2.0 | crates.io | runtime (proc macro) | yes |
| dashmap | 5.5.3 | MIT | crates.io | runtime | yes |
| dashmap | 6.2.1 | MIT | crates.io | runtime | yes |
| deadpool | 0.12.3 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| deadpool-runtime | 0.1.4 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| deranged | 0.5.8 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| digest | 0.10.7 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| digest | 0.11.3 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| displaydoc | 0.2.7 | MIT OR Apache-2.0 | crates.io | runtime (proc macro) | yes |
| either | 1.17.0 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| equivalent | 1.0.2 | Apache-2.0 OR MIT | crates.io | runtime | yes |
| errno | 0.3.14 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| error-chain | 0.12.4 | MIT/Apache-2.0 | crates.io | build | no |
| fastrand | 2.5.0 | Apache-2.0 OR MIT | crates.io | runtime | yes |
| find-msvc-tools | 0.1.9 | MIT OR Apache-2.0 | crates.io | build | yes |
| fixedbitset | 0.5.7 | MIT OR Apache-2.0 | crates.io | build | yes |
| float-cmp | 0.10.0 | MIT | crates.io | runtime | yes |
| fnv | 1.0.7 | Apache-2.0 / MIT | crates.io | runtime | yes |
| foldhash | 0.1.5 | Zlib | crates.io | build | yes |
| foldhash | 0.2.0 | Zlib | crates.io | runtime | yes |
| form_urlencoded | 1.2.2 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| fred | 10.1.0 | MIT; NVIDIA additions Apache-2.0 | vendored (`vendor/fred-10.1.0`) | runtime | yes |
| fred-macros | 0.1.0 | MIT | crates.io | runtime (proc macro) | yes |
| futures | 0.3.33 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| futures-channel | 0.3.33 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| futures-core | 0.3.33 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| futures-executor | 0.3.33 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| futures-io | 0.3.33 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| futures-macro | 0.3.33 | MIT OR Apache-2.0 | crates.io | runtime (proc macro) | yes |
| futures-sink | 0.3.33 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| futures-task | 0.3.33 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| futures-util | 0.3.33 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| generic-array | 0.14.7 | MIT | crates.io | runtime | yes |
| getrandom | 0.2.17 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| getrandom | 0.3.4 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| getrandom | 0.4.3 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| glob | 0.3.4 | MIT OR Apache-2.0 | crates.io | build | no |
| h2 | 0.4.19 | MIT | crates.io | runtime | yes |
| half | 2.7.1 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| hashbrown | 0.14.5 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| hashbrown | 0.15.5 | MIT OR Apache-2.0 | crates.io | build | yes |
| hashbrown | 0.17.1 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| heck | 0.5.0 | MIT OR Apache-2.0 | crates.io | build | yes |
| hermit-abi | 0.5.2 | MIT OR Apache-2.0 | crates.io | dev-test | no |
| hex | 0.4.3 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| hmac | 0.12.1 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| hmac | 0.13.0 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| home | 0.5.12 | MIT OR Apache-2.0 | crates.io | build | yes |
| http | 0.2.12 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| http | 1.5.0 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| http-body | 0.4.6 | MIT | crates.io | runtime | yes |
| http-body | 1.1.0 | MIT | crates.io | runtime | yes |
| http-body-util | 0.1.4 | MIT | crates.io | runtime | yes |
| httparse | 1.10.1 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| httpdate | 1.0.3 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| hybrid-array | 0.4.14 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| hyper | 1.11.0 | MIT | crates.io | runtime | yes |
| hyper-rustls | 0.27.9 | Apache-2.0 OR ISC OR MIT | crates.io | runtime | yes |
| hyper-timeout | 0.5.2 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| hyper-util | 0.1.20 | MIT | crates.io | runtime | yes |
| iana-time-zone | 0.1.65 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| iana-time-zone-haiku | 0.1.2 | MIT OR Apache-2.0 | crates.io | runtime | no |
| icu_collections | 2.2.0 | Unicode-3.0 | crates.io | runtime | yes |
| icu_locale_core | 2.2.0 | Unicode-3.0 | crates.io | runtime | yes |
| icu_normalizer | 2.2.0 | Unicode-3.0 | crates.io | runtime | yes |
| icu_normalizer_data | 2.2.0 | Unicode-3.0 | crates.io | runtime | yes |
| icu_properties | 2.2.0 | Unicode-3.0 | crates.io | runtime | yes |
| icu_properties_data | 2.2.0 | Unicode-3.0 | crates.io | runtime | yes |
| icu_provider | 2.2.0 | Unicode-3.0 | crates.io | runtime | yes |
| idna | 1.1.0 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| idna_adapter | 1.2.2 | Apache-2.0 OR MIT | crates.io | runtime | yes |
| indexmap | 2.14.0 | Apache-2.0 OR MIT | crates.io | runtime | yes |
| ipnet | 2.12.0 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| itertools | 0.13.0 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| itertools | 0.14.0 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| itoa | 1.0.18 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| js-sys | 0.3.103 | MIT OR Apache-2.0 | crates.io | runtime | no |
| lazy_static | 1.5.0 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| libc | 0.2.189 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| link-cplusplus | 1.0.12 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| linux-raw-sys | 0.4.15 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | crates.io | build | yes |
| linux-raw-sys | 0.12.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | crates.io | build | yes |
| litemap | 0.8.2 | Unicode-3.0 | crates.io | runtime | yes |
| live-edit | 0.2.0 | Apache-2.0 | workspace (`.`) | runtime | yes |
| live-edit-capabilities-proto | 0.1.0 | Apache-2.0 | workspace (`capabilities-proto`) | runtime | yes |
| live-edit-scaler | 0.1.0 | Apache-2.0 | workspace (`scaler`) | runtime | yes |
| lock_api | 0.4.14 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| log | 0.4.33 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| lru-slab | 0.1.2 | MIT OR Apache-2.0 OR Zlib | crates.io | runtime | yes |
| matchers | 0.2.0 | MIT | crates.io | runtime | yes |
| matchit | 0.7.3 | MIT AND BSD-3-Clause | crates.io | runtime | yes |
| matchit | 0.8.4 | MIT AND BSD-3-Clause | crates.io | runtime | yes |
| memchr | 2.8.3 | Unlicense OR MIT | crates.io | runtime | yes |
| mime | 0.3.17 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| mini-moka | 0.10.3 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| minimal-lexical | 0.2.1 | MIT/Apache-2.0 | crates.io | runtime | yes |
| mio | 1.2.2 | MIT | crates.io | runtime | yes |
| multimap | 0.10.1 | MIT OR Apache-2.0 | crates.io | build | yes |
| nom | 7.1.3 | MIT | crates.io | runtime | yes |
| nu-ansi-term | 0.50.3 | MIT | crates.io | runtime | yes |
| num-conv | 0.2.2 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| num-integer | 0.1.46 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| num-traits | 0.2.19 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| num_cpus | 1.17.0 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| once_cell | 1.21.4 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| oodf-ffi | 0.1.0 | Apache-2.0 | workspace (`oodf-ffi`) | runtime | yes |
| oorandom | 11.1.5 | MIT | crates.io | dev-test | yes |
| openssl-probe | 0.2.1 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| opentelemetry | 0.31.0 | Apache-2.0 | crates.io | runtime | yes |
| opentelemetry-http | 0.31.0 | Apache-2.0 | crates.io | runtime | yes |
| opentelemetry-otlp | 0.31.1 | Apache-2.0 | crates.io | runtime | yes |
| opentelemetry-prometheus-text-exporter | 0.2.1 | Apache-2.0 | crates.io | runtime | yes |
| opentelemetry-proto | 0.31.0 | Apache-2.0 | crates.io | runtime | yes |
| opentelemetry_sdk | 0.31.0 | Apache-2.0 | crates.io | runtime | yes |
| ordered-float | 4.6.0 | MIT | crates.io | runtime | yes |
| outref | 0.5.2 | MIT | crates.io | runtime | yes |
| page_size | 0.6.0 | MIT/Apache-2.0 | crates.io | dev-test | yes |
| parking_lot | 0.12.5 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| parking_lot_core | 0.9.12 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| percent-encoding | 2.3.2 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| petgraph | 0.8.3 | MIT OR Apache-2.0 | crates.io | build | yes |
| pin-project | 1.1.13 | Apache-2.0 OR MIT | crates.io | runtime | yes |
| pin-project-internal | 1.1.13 | Apache-2.0 OR MIT | crates.io | runtime (proc macro) | yes |
| pin-project-lite | 0.2.17 | Apache-2.0 OR MIT | crates.io | runtime | yes |
| pin-utils | 0.1.0 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| potential_utf | 0.1.5 | Unicode-3.0 | crates.io | runtime | yes |
| powerfmt | 0.2.0 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| ppv-lite86 | 0.2.21 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| prettyplease | 0.2.37 | MIT OR Apache-2.0 | crates.io | build | yes |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| prometheus | 0.14.0 | Apache-2.0 | crates.io | dev-test | yes |
| proptest | 1.11.0 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| prost | 0.14.4 | Apache-2.0 | crates.io | runtime | yes |
| prost-build | 0.14.4 | Apache-2.0 | crates.io | build | yes |
| prost-derive | 0.14.4 | Apache-2.0 | crates.io | runtime (proc macro) | yes |
| prost-types | 0.14.4 | Apache-2.0 | crates.io | runtime | yes |
| pulldown-cmark | 0.9.6 | MIT | crates.io | build | no |
| pulldown-cmark | 0.13.4 | MIT | crates.io | build | yes |
| pulldown-cmark-to-cmark | 22.0.0 | Apache-2.0 | crates.io | build | yes |
| quick-error | 1.2.3 | MIT/Apache-2.0 | crates.io | dev-test | yes |
| quinn | 0.11.11 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| quinn-proto | 0.11.19 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| quinn-udp | 0.5.15 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| quote | 1.0.47 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| r-efi | 5.3.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later | crates.io | runtime | no |
| r-efi | 6.0.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later | crates.io | runtime | no |
| rand | 0.8.7 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| rand | 0.9.5 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| rand | 0.10.2 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| rand_chacha | 0.3.1 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| rand_chacha | 0.9.0 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| rand_core | 0.6.4 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| rand_core | 0.9.5 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| rand_core | 0.10.1 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| rand_pcg | 0.10.2 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| rand_xorshift | 0.4.0 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| redis-protocol | 6.0.0 | MIT | crates.io | runtime | yes |
| redox_syscall | 0.5.18 | MIT | crates.io | runtime | no |
| regex | 1.13.1 | MIT OR Apache-2.0 | crates.io | build | yes |
| regex-automata | 0.4.16 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| regex-lite | 0.1.9 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| regex-syntax | 0.8.11 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| reqwest | 0.12.28 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| ring | 0.17.14 | Apache-2.0 AND ISC | crates.io | runtime | yes |
| rustc-hash | 2.1.3 | Apache-2.0 OR MIT | crates.io | runtime | yes |
| rustc_version | 0.4.1 | MIT OR Apache-2.0 | crates.io | build | yes |
| rustix | 0.38.44 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | crates.io | build | yes |
| rustix | 1.1.4 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | crates.io | build | yes |
| rustls | 0.23.46 | Apache-2.0 OR ISC OR MIT | crates.io | runtime | yes |
| rustls-native-certs | 0.8.4 | Apache-2.0 OR ISC OR MIT | crates.io | runtime | yes |
| rustls-pki-types | 1.15.1 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| rustls-webpki | 0.103.15 | ISC | crates.io | runtime | yes |
| rustversion | 1.0.23 | MIT OR Apache-2.0 | crates.io | runtime (proc macro) | yes |
| rusty-fork | 0.3.1 | MIT/Apache-2.0 | crates.io | dev-test | yes |
| ryu | 1.0.23 | Apache-2.0 OR BSL-1.0 | crates.io | runtime | yes |
| same-file | 1.0.6 | Unlicense/MIT | crates.io | build | yes |
| schannel | 0.1.29 | MIT | crates.io | runtime | no |
| scopeguard | 1.2.0 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| scratch | 1.0.9 | MIT OR Apache-2.0 | crates.io | build | yes |
| security-framework | 3.7.0 | MIT OR Apache-2.0 | crates.io | runtime | no |
| security-framework-sys | 2.17.0 | MIT OR Apache-2.0 | crates.io | runtime | no |
| semver | 1.0.28 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| serde | 1.0.229 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| serde_core | 1.0.229 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| serde_derive | 1.0.229 | MIT OR Apache-2.0 | crates.io | runtime (proc macro) | yes |
| serde_json | 1.0.151 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| serde_path_to_error | 0.1.20 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| serde_urlencoded | 0.7.1 | MIT/Apache-2.0 | crates.io | runtime | yes |
| sha-1 | 0.10.1 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| sha1 | 0.10.7 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| sha2 | 0.10.9 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| sha2 | 0.11.0 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| sharded-slab | 0.1.7 | MIT | crates.io | runtime | yes |
| shlex | 1.3.0 | MIT OR Apache-2.0 | crates.io | build | yes |
| signal-hook-registry | 1.4.8 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| skeptic | 0.13.7 | MIT/Apache-2.0 | crates.io | build | no |
| slab | 0.4.12 | MIT | crates.io | runtime | yes |
| smallvec | 1.15.2 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| smartstring | 1.0.1 | MPL-2.0+ | crates.io | runtime | yes |
| socket2 | 0.5.10 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| socket2 | 0.6.5 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| stable_deref_trait | 1.2.1 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| static_assertions | 1.1.0 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| strsim | 0.11.1 | MIT | crates.io | dev-test | yes |
| subtle | 2.6.1 | BSD-3-Clause | crates.io | runtime | yes |
| syn | 2.0.119 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| syn | 3.0.3 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| sync_wrapper | 1.0.2 | Apache-2.0 | crates.io | runtime | yes |
| synstructure | 0.13.2 | MIT | crates.io | runtime | yes |
| tagptr | 0.2.0 | MIT/Apache-2.0 | crates.io | runtime | yes |
| temp-env | 0.3.6 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| tempfile | 3.27.0 | MIT OR Apache-2.0 | crates.io | build | yes |
| termcolor | 1.4.1 | Unlicense OR MIT | crates.io | build | yes |
| thiserror | 2.0.19 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| thiserror-impl | 2.0.19 | MIT OR Apache-2.0 | crates.io | runtime (proc macro) | yes |
| thread_local | 1.1.10 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| tikv-jemalloc-sys | 0.6.1+5.3.0-1-ge13ca993e8ccb9ba9847cc330696e02839f328f7 | MIT/Apache-2.0 | crates.io | runtime | yes |
| tikv-jemallocator | 0.6.1 | MIT/Apache-2.0 | crates.io | runtime | yes |
| time | 0.3.54 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| time-core | 0.1.9 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| time-macros | 0.2.32 | MIT OR Apache-2.0 | crates.io | runtime (proc macro) | yes |
| tinystr | 0.8.3 | Unicode-3.0 | crates.io | runtime | yes |
| tinytemplate | 1.2.1 | Apache-2.0 OR MIT | crates.io | dev-test | yes |
| tinyvec | 1.12.0 | Zlib OR Apache-2.0 OR MIT | crates.io | runtime | yes |
| tinyvec_macros | 0.1.1 | MIT OR Apache-2.0 OR Zlib | crates.io | runtime | yes |
| tokio | 1.53.1 | MIT | crates.io | runtime | yes |
| tokio-macros | 2.7.2 | MIT | crates.io | runtime (proc macro) | yes |
| tokio-rustls | 0.26.4 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| tokio-stream | 0.1.19 | MIT | crates.io | runtime | yes |
| tokio-util | 0.7.19 | MIT | crates.io | runtime | yes |
| tonic | 0.14.6 | MIT | crates.io | runtime | yes |
| tonic-build | 0.14.6 | MIT | crates.io | build | yes |
| tonic-health | 0.14.6 | MIT | crates.io | runtime | yes |
| tonic-prost | 0.14.6 | MIT | crates.io | runtime | yes |
| tonic-prost-build | 0.14.6 | MIT | crates.io | build | yes |
| tonic-reflection | 0.14.6 | MIT | crates.io | runtime | yes |
| tonic-types | 0.14.6 | MIT | crates.io | runtime | yes |
| tower | 0.5.3 | MIT | crates.io | runtime | yes |
| tower-http | 0.6.11 | MIT | crates.io | runtime | yes |
| tower-layer | 0.3.3 | MIT | crates.io | runtime | yes |
| tower-service | 0.3.3 | MIT | crates.io | runtime | yes |
| tracing | 0.1.44 | MIT | crates.io | runtime | yes |
| tracing-attributes | 0.1.31 | MIT | crates.io | runtime (proc macro) | yes |
| tracing-core | 0.1.36 | MIT | crates.io | runtime | yes |
| tracing-log | 0.2.0 | MIT | crates.io | runtime | yes |
| tracing-opentelemetry | 0.32.1 | MIT | crates.io | runtime | yes |
| tracing-serde | 0.2.0 | MIT | crates.io | runtime | yes |
| tracing-subscriber | 0.3.23 | MIT | crates.io | runtime | yes |
| triomphe | 0.1.16 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| try-lock | 0.2.5 | MIT | crates.io | runtime | yes |
| typenum | 1.20.1 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| unarray | 0.1.4 | MIT OR Apache-2.0 | crates.io | dev-test | yes |
| unicase | 2.9.0 | MIT OR Apache-2.0 | crates.io | build | yes |
| unicode-ident | 1.0.24 | (MIT OR Apache-2.0) AND Unicode-3.0 | crates.io | runtime | yes |
| unicode-width | 0.2.2 | MIT OR Apache-2.0 | crates.io | build | yes |
| untrusted | 0.9.0 | ISC | crates.io | runtime | yes |
| url | 2.5.8 | MIT OR Apache-2.0 | crates.io | runtime | yes |
| urlencoding | 2.1.3 | MIT | crates.io | runtime | yes |
| utf8_iter | 1.0.4 | Apache-2.0 OR MIT | crates.io | runtime | yes |
| uuid | 1.24.0 | Apache-2.0 OR MIT | crates.io | runtime | yes |
| valuable | 0.1.1 | MIT | crates.io | runtime | no |
| version_check | 0.9.5 | MIT/Apache-2.0 | crates.io | build | yes |
| vsimd | 0.8.0 | MIT | crates.io | runtime | yes |
| wait-timeout | 0.2.1 | MIT/Apache-2.0 | crates.io | dev-test | yes |
| walkdir | 2.5.0 | Unlicense/MIT | crates.io | build | yes |
| want | 0.3.1 | MIT | crates.io | runtime | yes |
| wasi | 0.11.1+wasi-snapshot-preview1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | crates.io | runtime | no |
| wasip2 | 1.0.4+wasi-0.2.12 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | crates.io | runtime | no |
| wasm-bindgen | 0.2.126 | MIT OR Apache-2.0 | crates.io | runtime | no |
| wasm-bindgen-futures | 0.4.76 | MIT OR Apache-2.0 | crates.io | runtime | no |
| wasm-bindgen-macro | 0.2.126 | MIT OR Apache-2.0 | crates.io | runtime (proc macro) | no |
| wasm-bindgen-macro-support | 0.2.126 | MIT OR Apache-2.0 | crates.io | runtime | no |
| wasm-bindgen-shared | 0.2.126 | MIT OR Apache-2.0 | crates.io | runtime | no |
| web-sys | 0.3.103 | MIT OR Apache-2.0 | crates.io | runtime | no |
| web-time | 1.1.0 | MIT OR Apache-2.0 | crates.io | runtime | no |
| webpki-roots | 1.0.9 | CDLA-Permissive-2.0 | crates.io | runtime | yes |
| which | 6.0.3 | MIT | crates.io | build | yes |
| winapi | 0.3.9 | MIT/Apache-2.0 | crates.io | dev-test | no |
| winapi-i686-pc-windows-gnu | 0.4.0 | MIT/Apache-2.0 | crates.io | dev-test | no |
| winapi-util | 0.1.11 | Unlicense OR MIT | crates.io | build | no |
| winapi-x86_64-pc-windows-gnu | 0.4.0 | MIT/Apache-2.0 | crates.io | dev-test | no |
| windows-core | 0.62.2 | MIT OR Apache-2.0 | crates.io | runtime | no |
| windows-implement | 0.60.2 | MIT OR Apache-2.0 | crates.io | runtime (proc macro) | no |
| windows-interface | 0.59.3 | MIT OR Apache-2.0 | crates.io | runtime (proc macro) | no |
| windows-link | 0.2.1 | MIT OR Apache-2.0 | crates.io | runtime | no |
| windows-result | 0.4.1 | MIT OR Apache-2.0 | crates.io | runtime | no |
| windows-strings | 0.5.1 | MIT OR Apache-2.0 | crates.io | runtime | no |
| windows-sys | 0.52.0 | MIT OR Apache-2.0 | crates.io | runtime | no |
| windows-sys | 0.59.0 | MIT OR Apache-2.0 | crates.io | build | no |
| windows-sys | 0.61.2 | MIT OR Apache-2.0 | crates.io | runtime | no |
| windows-targets | 0.52.6 | MIT OR Apache-2.0 | crates.io | runtime | no |
| windows_aarch64_gnullvm | 0.52.6 | MIT OR Apache-2.0 | crates.io | runtime | no |
| windows_aarch64_msvc | 0.52.6 | MIT OR Apache-2.0 | crates.io | runtime | no |
| windows_i686_gnu | 0.52.6 | MIT OR Apache-2.0 | crates.io | runtime | no |
| windows_i686_gnullvm | 0.52.6 | MIT OR Apache-2.0 | crates.io | runtime | no |
| windows_i686_msvc | 0.52.6 | MIT OR Apache-2.0 | crates.io | runtime | no |
| windows_x86_64_gnu | 0.52.6 | MIT OR Apache-2.0 | crates.io | runtime | no |
| windows_x86_64_gnullvm | 0.52.6 | MIT OR Apache-2.0 | crates.io | runtime | no |
| windows_x86_64_msvc | 0.52.6 | MIT OR Apache-2.0 | crates.io | runtime | no |
| winsafe | 0.0.19 | MIT | crates.io | build | no |
| wiremock | 0.6.5 | MIT/Apache-2.0 | crates.io | dev-test | yes |
| wit-bindgen | 0.57.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | crates.io | runtime | no |
| writeable | 0.6.3 | Unicode-3.0 | crates.io | runtime | yes |
| xmlparser | 0.13.6 | MIT/Apache-2.0 | crates.io | runtime | yes |
| yoke | 0.8.3 | Unicode-3.0 | crates.io | runtime | yes |
| yoke-derive | 0.8.2 | Unicode-3.0 | crates.io | runtime (proc macro) | yes |
| zerocopy | 0.8.55 | BSD-2-Clause OR Apache-2.0 OR MIT | crates.io | runtime | yes |
| zerocopy-derive | 0.8.55 | BSD-2-Clause OR Apache-2.0 OR MIT | crates.io | runtime (proc macro) | yes |
| zerofrom | 0.1.8 | Unicode-3.0 | crates.io | runtime | yes |
| zerofrom-derive | 0.1.7 | Unicode-3.0 | crates.io | runtime (proc macro) | yes |
| zeroize | 1.9.0 | Apache-2.0 OR MIT | crates.io | runtime | yes |
| zerotrie | 0.2.4 | Unicode-3.0 | crates.io | runtime | yes |
| zerovec | 0.11.6 | Unicode-3.0 | crates.io | runtime | yes |
| zerovec-derive | 0.11.3 | Unicode-3.0 | crates.io | runtime (proc macro) | yes |
| zmij | 1.0.23 | MIT | crates.io | runtime | yes |
