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

# Source and dependency inventory

This file records what each Live Edit release artifact contains, how its files are licensed, which third-party code it depends on or carries, and which cryptography the service uses.
It is the license and provenance record for anyone who redistributes this source or the artifacts built from it.
[RUST-DEPENDENCY-LICENSES.md](RUST-DEPENDENCY-LICENSES.md) gives the license of every Rust package in the root `Cargo.lock`, as of the lockfile whose SHA-256 it records.

## 1. Release scopes

Each artifact below is built from this source tree; the notices it carries are listed with it.

### Source tree

The published source tree is this repository without its internal planning, CI configuration, design documents and internal tooling.
Its notices are at the root:

- `LICENSE.txt`: the Apache License 2.0, verbatim.
- `NOTICE`: NVIDIA's attribution, the license split of the vendored fred crate, and where the copied sources' provenance is recorded.
- `ATTRIBUTIONS.md`: the license texts of the Rust crates the service is built from (section 3).
- `oss/OSS_LICENSE.txt`: a second copy of the Apache License 2.0; `oss/SECURITY.md`: how to report a vulnerability.
- This file and `oss/RUST-DEPENDENCY-LICENSES.md`.

Its lockfiles are the workspace's `Cargo.lock` and `oodf-ffi/fuzz/Cargo.lock`, which locks what the fuzz harness is built from.
CI audits both against the RustSec advisory database.

A repository that imports the tree carries these files with it (section 6).

### Service container image

Built by `Dockerfile`.
A Debian Rust stage compiles the workspace, then `tools/package_oss.py` collects the third-party sources and notices.
The runtime stage is NVIDIA's distroless `cc` base image, which supplies its own notices.
The image adds three paths:

| Path | Contents |
|---|---|
| `/app/live-edit` | The service binary. The replica autoscaler is the same binary, run as `live-edit scaler`. |
| `/licenses/` | `LICENSE.txt`, `NOTICE` and `ATTRIBUTIONS.md`; the FlatBuffers license and header notice; the Rust standard library's copyright file and licenses; the GCC runtime's copyright file and the GPL-3.0 text; `SOURCE-MANIFEST.json` (the build inputs: binary and `Cargo.lock` digests, `rustc`, FlatBuffers and GCC versions); `SOURCES.md`, which describes `/vendor.tar.gz`; `SHA256SUMS`. |
| `/vendor.tar.gz` | The source of the third-party code compiled into the binary: the locked Cargo dependencies (`sources/crates/<name>-<version>`), the patched fred tree without its `Cargo.lock`, with its patch, its provenance record and its upstream-file manifest `fred-10.1.0.SHA256SUMS` (`sources/patched-crates/`), the FlatBuffers checkout, the Rust standard library sources with their locked dependencies, and the Debian GCC source package of the statically linked C++ runtime. |

`/vendor.tar.gz` is a superset: it also holds the build, test and other-platform crates of the lockfile.
Its own files leave out the lockfiles third-party projects keep for their own development, which pin versions the binary was not built from; the only lockfiles among them are the workspace's `Cargo.lock` and the Rust standard library's.
The FlatBuffers `git archive`, the standard library's `.crate` files and the GCC source files are kept as published, with any lockfiles inside them.
It carries no NVIDIA application source; NVIDIA's changes to fred, in the patched tree, are the only NVIDIA-authored code in it.

### API archive

The API archive is a zip file for client developers.
It contains:

- the LiveService protocol files, `proto/nvidia/omniverse/live/`;
- `CHANGELOG.md`, `LICENSE.txt`, `NOTICE`, `OSS_LICENSE.txt` and `SECURITY.md`;
- `PRODUCT_TERMS_OMNIVERSE.txt`, a reference copy of the NVIDIA Omniverse product-specific terms, which the source tree does not carry.

It carries no third-party code, so it has no `ATTRIBUTIONS.md`, and no conformance suite: the suite ships as the conformance-test image (below).

### Conformance-test image

Built by `Dockerfile.conformance`.
Its builder steps are the service `Dockerfile`'s, verbatim, and it compiles only the suite's crate, `live-edit-conformance` (`conformance_tests/`); then `tools/package_oss.py --binary live-edit-conformance` collects the third-party sources and notices.
Its runtime stage is the service image's distroless `cc` base, the same tag and digest.
A test fails the CI pipeline when the two files' builder steps or runtime bases differ.
The image adds these paths:

| Path | Contents |
|---|---|
| `/app/live-edit-conformance` | The suite binary, NVIDIA code with the suite's fixtures, which are NVIDIA data, built in. |
| `/licenses/` | The same files as the service image's `/licenses/`, with `ATTRIBUTIONS.md` byte-identical; `SOURCE-MANIFEST.json` and `SHA256SUMS` bind them to the suite binary, and `SOURCES.md` has no jemalloc paragraph, since the suite binary does not link jemalloc. |
| `/vendor.tar.gz` | The same source set as the service image's: the whole `Cargo.lock`, FlatBuffers, the Rust standard library and the GCC source package. Like the service image's, its own files' only lockfiles are the workspace's `Cargo.lock` and the standard library's, and the archives it carries keep any lockfiles inside them. |
| `/opt/local-test-token.jwt` | The public unsigned test token, which is not a credential. |
| `/reports/` | An empty directory owned by UID/GID 65532, the image's user, for the suite's reports. |

Every third-party Rust crate compiled into the suite binary is one the service binary also compiles in, at the same version; `conformance_tests/tests/dependency_fence.rs` fails otherwise.
With the shared builder and base, one third-party component set covers both images.

### Helm charts

Two charts are packaged from `helm/`:

- `live-edit-service` deploys the service and its autoscaler. It includes a smoke-test Job whose test sources and generated Python gRPC stubs (`files/`) the chart mounts into a smoke-test base image it names in `smokeTest.image`.
- `live-edit-conformance-tests` runs the conformance-test image as a Job, whose command is `/app/live-edit-conformance`.

Every chart file carries the NVIDIA Apache-2.0 header.
The charts hold no third-party code, carry no separate notice files, and contain no container images: they name the images they deploy, which are distributed separately.

## 2. How files are licensed

- **NVIDIA-authored files** carry an SPDX header naming NVIDIA as the copyright holder and `Apache-2.0` as the license, written in the file's comment syntax (an HTML comment at the top of a Markdown file).
  `python3 tools/license_headers.py check` enforces this.
- **Files with no comment syntax** (binary and JSON fixtures, lockfiles, the test token, license and notice texts, the fred patch and checksum records) are listed in `tools/license-classification.json`, each with its class, owner, license and provenance.
  The same check requires an entry for every such file.
- **Preserved upstream files** in `vendor/fred-10.1.0/` are fred 10.1.0 as published on crates.io, under fred's MIT license with fred's own notices.
  `vendor/fred-10.1.0.SHA256SUMS` records the published bytes of every upstream file.
  The published source tree lacks two of them, the crate's `Cargo.lock` and `.vscode/settings.json`, which no build reads.
  The container images' `/vendor.tar.gz` copy of the tree lacks the crate's `Cargo.lock` too; the manifest still lists it, as the record of the published release.
  `scripts/check-vendored-fred.sh` rebuilds the tree from the published crate and the patch, and fails on any difference.
- **NVIDIA's modifications** to 23 of fred's files are licensed under the MIT license, so each modified file stays under MIT alone; every hunk of the patch that edits one of them carries a `live-edit patch` comment on at least one of its added lines.
- **NVIDIA's added files** in the fred tree, under `vendor/fred-10.1.0/src/live_edit/`, are Apache-2.0 and carry NVIDIA's header.
  `vendor/fred-10.1.0.SOURCE.md` records the source, the patch and this license split; the tree as a whole is `MIT AND Apache-2.0`.
- **The OODF schema**, `oodf-ffi/schema/OmniObjectDelta.fbs`, is Apache-2.0 and carries NVIDIA's header, which replaced the original NVIDIA notice of the upstream file; the rest of the file is upstream's, byte for byte.
  `oodf-ffi/schema/SOURCE.md` records the provenance, the upstream and adapted hashes and this notice-only relicense.

## 3. Dependencies

### Rust crates

`ATTRIBUTIONS.md` reproduces the license text of every crate that is a normal (non-build, non-test) dependency of the workspace, on every platform.
`tools/generate_attributions.py` generates it from `Cargo.lock` and the locked crate archives.
`RUST-DEPENDENCY-LICENSES.md` lists every package the root `Cargo.lock` locks with its license, source and scope, as of the lockfile whose SHA-256 it records; it is not kept current.

**smartstring is the one weak-copyleft crate compiled into the service.**

- `smartstring` 1.0.1 is licensed under MPL-2.0; its manifest declares `MPL-2.0+`, which `ATTRIBUTIONS.md` records as `MPL-2.0`.
- It comes unmodified from crates.io, with the checksum `Cargo.lock` records.
- It is a normal dependency of `opentelemetry-prometheus-text-exporter` 0.2.1, which `live-edit` and `live-edit-scaler` use to serve `/metrics`, so it is compiled into the service binary.
- Its license text is the `## smartstring` section of `ATTRIBUTIONS.md`, and its source is `sources/crates/smartstring-1.0.1` in the image's `/vendor.tar.gz`.

No other crate in the root `Cargo.lock` declares MPL, LGPL, GPL, EPL or another copyleft license, with one exception that is never built: `r-efi` 5.3.0 and 6.0.0 offer `MIT OR Apache-2.0 OR LGPL-2.1-or-later`, and are locked only for UEFI targets.
`webpki-roots`, the Mozilla CA list compiled in as the REST clients' trust roots, is `CDLA-Permissive-2.0`, a permissive data license.

### C and C++ code compiled into the service

The conformance-test image's binary carries the same components, with the same notices and sources in that image, except jemalloc.

| Component | License | Notice and source in the service image |
|---|---|---|
| FlatBuffers 24.3.25 runtime headers | Apache-2.0 | `/licenses/FLATBUFFERS-LICENSE.txt`, `/licenses/FLATBUFFERS-NOTICE.txt`; `flatbuffers.tar` in `/vendor.tar.gz` |
| GCC C++ runtime (`libstdc++`, statically linked) | GPL-3.0 with the GCC Runtime Library Exception | `/licenses/GCC-COPYRIGHT.txt`, `/licenses/GPL-3.txt`; the Debian GCC source package in `/vendor.tar.gz` |
| Rust standard library | MIT OR Apache-2.0 | `/licenses/RUST-COPYRIGHT.html`, `/licenses/rust-licenses/`; the `rust-src` sources in `/vendor.tar.gz` |
| jemalloc, built by the `tikv-jemalloc-sys` crate | BSD-2-Clause | `ATTRIBUTIONS.md` under `tikv-jemalloc-sys`; the crate's `jemalloc/` directory in `/vendor.tar.gz` |
| `ring`'s C and assembly code (the TLS crypto provider) | Apache-2.0 AND ISC | `ATTRIBUTIONS.md` under `ring`; the crate in `/vendor.tar.gz` |

### Build and CI tools

These tools build, generate, test or check the code.
None is distributed in a release artifact; where a tool's output is compiled in, the table says so.

| Tool | Used for | License |
|---|---|---|
| Rust toolchain 1.94 (`rustc`, `cargo`, `rust-src`) | Compiling the workspace; the standard library is compiled in (above) | MIT OR Apache-2.0 |
| FlatBuffers `flatc` 24.3.25, built from source | Generating the OODF C++ code at build time | Apache-2.0 |
| GCC (`g++`) | Compiling the C++ library | GPL-3.0-or-later (its runtime library is compiled in, above) |
| `clang` and `mold` | Linking on Linux | Apache-2.0 WITH LLVM-exception; MIT |
| CMake | Building `flatc` | BSD-3-Clause |
| `protoc` (Debian `protobuf-compiler`) | Generating Rust gRPC code from `proto/` at build time | BSD-3-Clause |
| `buf`, with `protoc-gen-prost`, `protoc-gen-tonic` and `protoc-gen-prost-crate` | Regenerating `capabilities-proto/` | Apache-2.0 |
| `grpcio-tools` 1.76.0 | Regenerating the committed Python gRPC stubs | Apache-2.0 |
| `uv` 0.8.22 | Installing the Python packages of CI's pipeline-structure tests, which are not published | MIT OR Apache-2.0 |
| `cargo-binstall` 1.10.13 | Installing the CI tools below | GPL-3.0-only |
| `cargo-audit` 0.22.2 | RustSec advisory check | Apache-2.0 OR MIT |
| `cargo-deny` | License allowlist and advisory check | MIT OR Apache-2.0 |
| `dd-rust-license-tool` 1.0.6 | The license list `ATTRIBUTIONS.md` is generated from | MIT |
| `cargo-nextest` | Test runner | Apache-2.0 OR MIT |
| `cargo-fuzz` | Fuzzing the OODF parser | MIT OR Apache-2.0 |
| `ruff` 0.12.12 | Python lint | MIT |

Each license was read from the tool's crates.io or PyPI record for the version named, or from its source repository's license file.

## 4. Code copied into the published source

- **fred 10.1.0, patched**, in `vendor/fred-10.1.0/` (section 2).
  The crate archive's SHA-256 is `3a7b2fd0f08b23315c13b6156f971aeedb6f75fb16a29ac1872d2eabccc1490e`.
  The patch is `vendor/fred-10.1.0.patch`, and `vendor/fred-10.1.0.UPSTREAM.md` describes its changes for fred's maintainers.
- **The OODF schema**, `oodf-ffi/schema/OmniObjectDelta.fbs`, an NVIDIA file copied from the NVIDIA service that owns it, with its notice replaced by NVIDIA's Apache-2.0 header (section 2); `oodf-ffi/schema/SOURCE.md` records its provenance, the upstream and adapted hashes and the relicense.
- **NVIDIA protocol files from other NVIDIA services**: `proto/nvidia/omniverse/storage/fileobject/v1beta/` (the storage service's file-object API) and `proto/nvidia/omniverse/userinfo/v1alpha/` (the user-info service API).
  Both carry NVIDIA's Apache-2.0 header; the service's build generates its client code from them.
- **Generated code committed to the tree**, all generated from NVIDIA protocol files and carrying NVIDIA's header:
  - `capabilities-proto/src/`, generated by `buf` and the `protoc-gen-prost` plugins from the Discovery service's published capabilities API (`scripts/regenerate-capabilities-proto.sh`);
  - `helm/live-edit-service/files/generated/nvidia/omniverse/live/v1alpha/live_service_pb2*.py`, generated by `grpcio-tools` from `live_service.proto`.

  The protobuf license states that code its compiler generates is owned by the owner of the input file.
- **Test fixtures.** `tools/license-classification.json` classifies each data fixture by its `class` field:
  - `captured-data`: bytes recorded from a C++ reference service or a Kit client, such as `tools/fixtures/gze_inbound_deltas.json`, `oodf-ffi/tests/fixtures/postman_chunk.bin` and three of the `.oodf` files in `tests/fixtures/oodf/`;
  - `captured-derived`: the `tests/fixtures/tail_lineage/` files, which a generator produces by replaying a captured fixture through the service's commit path;
  - `synthetic-data`: fixtures the in-tree generators write, such as the other `.oodf` files.
    The conformance suite builds its synthetic fixtures when it runs, from definitions in its source, so none is committed.

  Each `.oodf`, `.ompf` and `.letl` fixture has a Markdown sidecar beside it recording where it came from; `tests/fixtures/oodf/CAPTURE.md` describes the capture procedure.

Outside `vendor/`, no file carries a third-party copyright notice, besides `ATTRIBUTIONS.md`, which reproduces the third-party license texts (section 3).
Besides `ATTRIBUTIONS.md`'s, the only non-NVIDIA copyright lines there are the appendix template line of the Apache License texts, fred's attribution in `NOTICE`, and invented or quoted example notices that the published tests of the header and attribution tools use as inputs.
No file states that it was copied or adapted from third-party code.

## 5. Cryptography

### Transport security

The service's listeners accept plaintext only: gRPC on port 50001 and HTTP (`/metrics`) on port 9090.
The service has no server TLS configuration, so client TLS terminates at an ingress in front of it.
The service chart can create a Contour `HTTPProxy` that terminates TLS with a certificate from a Kubernetes Secret, optionally issued by a cert-manager `Certificate`, and forwards to the service over plaintext HTTP/2 (h2c).

The service opens TLS connections only as a client, and only where the configured address asks for it:

| Outbound connection | TLS when | Client and trust roots |
|---|---|---|
| Valkey backing store | The URL scheme is `rediss://` or `valkeys://` (with any of their cluster or sentinel forms) | fred; the system store |
| Authorization (storage service) and user-info service, gRPC | The endpoint is `https://` or its port is 443 | tonic; the system store |
| Authorization and user-info service, REST | The endpoint is `https://` | reqwest; the Mozilla CA list compiled in (`webpki-roots`) |
| Storage API checkpoint adapter, and the pre-signed storage URLs it is redirected to | The URL is `https://` | reqwest; the compiled-in Mozilla CA list |
| AWS STS, for IAM backing-store authentication only | Always | The AWS SDK's HTTP client; the system store |
| Kubernetes API, from the autoscaler (`live-edit scaler`) only | Always | reqwest; only the pod service account's CA certificate |
| OTLP trace and metric export | The endpoint is `https://` | tonic; the system store |
| Autoscaler polls of service replicas | Never: plaintext HTTP inside the cluster | reqwest |

"The system store" is the image's system certificate store or, when `SSL_CERT_FILE` or `SSL_CERT_DIR` is set, only the locations they name.
Every TLS connection uses rustls 0.23 with the `ring` crypto provider.
`Cargo.lock` contains no OpenSSL binding (`openssl` or `openssl-sys`), native-tls or `aws-lc` package; `openssl-probe`, through `rustls-native-certs`, only locates the system certificate files.
The service image's base layer carries its distribution's own OpenSSL libraries, which the service binary does not link and never loads.
`scripts/check-crypto-provider-invariant.sh`, run by the `supply-chain:crypto-provider-invariant` CI job, fails if the lockfile gains `aws-lc-rs` or loses `ring`, or if fred's TLS feature routing changes; `deny.toml` bans the `aws-lc` crates as a second check.

### Cryptographic operations

| Operation | Algorithm | Implementation | Purpose |
|---|---|---|---|
| TLS client | TLS, `ring` primitives | rustls, `ring` | The outbound connections above |
| External-value content address and upload check | SHA-256 over the SHA-256 digests of the value's 1 MiB blocks | `sha2` | Identifying and verifying external values |
| Re-send detection for `operation_id` | SHA-256 of the layer name and the uploaded bytes | `sha2` | Telling a re-send from a reused id |
| Auth-cache keys | HMAC-SHA256 under a 32-byte per-process key from the operating system's random generator | `hmac`, `sha2`, `rand` | Caching authorization results without holding raw bearer tokens |
| Session-checkpoint incarnation token | 128 bits from the operating system's random generator, written as 32 hex characters | `rand` | Telling a session's checkpoint record recreated after a backing-store wipe from the one before it, so a stale checkpoint publish is refused; an identifier, not a security control |
| Credential-file change detection, checkpoint-store identity | SHA-256, truncated, kept in memory or used as an identifier | `sha2` | Equality checks only |
| IAM backing-store authentication | AWS Signature Version 4 (HMAC-SHA256) pre-signed token | `aws-sigv4` | The Valkey `AUTH` password for Amazon ElastiCache and MemoryDB |
| Script and state digests | SHA-1 | `sha1`, and fred's `sha-1` feature | Valkey Lua script digests for `EVALSHA`, and digests that must equal Valkey's `redis.sha1hex`; identifiers, not a security control |
| JWT payload decoding | Base64url decoding, no signature verification | `base64`, `serde_json` | Reading display-name claims; the authorization service decides whether a token is valid |

The service implements no cryptographic algorithm of its own, and does not encrypt the data it writes to the backing store or the checkpoint store; in transit, that data is protected only by the TLS connections above.

The conformance-test binary uses the same crates: rustls with the `ring` provider for TLS, through tonic to an `https://` service endpoint (the system store) and through reqwest to the optional Storage API endpoint (the compiled-in Mozilla CA list), a `CONFORMANCE_CA_FILE` bundle replacing both; and SHA-256 (`sha2`) for external-value content addresses and its fixtures' pinned hashes.

## 6. For repositories that import this tree

A repository that imports this source tree, whole or as a subdirectory, carries these files with it:

- `NOTICE`, `LICENSE.txt` and `ATTRIBUTIONS.md`;
- this file and `oss/RUST-DEPENDENCY-LICENSES.md`;
- `tools/license-classification.json`, with `tools/license_headers.py`, which checks the tree against it;
- the provenance records section 4 names: `vendor/fred-10.1.0.SOURCE.md`, `vendor/fred-10.1.0.SHA256SUMS`, `vendor/fred-10.1.0.patch` and `oodf-ffi/schema/SOURCE.md`.

The importing repository's own inventory covers its own code and dependencies, and links these files for the imported tree.
