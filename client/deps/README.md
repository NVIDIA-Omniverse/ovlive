<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dependency manifests

This directory holds the packman manifests for ovlive's native transport dependencies and records the host SDKs each build profile consumes. It is for engineers who provision a build worker; the [build guide](../docs/ovlive-build.md) explains how the provisioned roots are passed to the build.

## Transport dependencies

`live-services-target.packman.xml` lists the statically linked gRPC closure. License texts are in [`THIRD_PARTY_LICENSES.txt`](../THIRD_PARTY_LICENSES.txt).

| Package | Version prefix | License |
| --- | --- | --- |
| grpc | 1.65.4-1 | Apache-2.0 AND BSD-3-Clause AND MPL-2.0 |
| protobuf | 5.26.0-2 | BSD-3-Clause |
| abseil | 20240116.1-1 | Apache-2.0 |
| re2 | 20230301-1 | BSD-3-Clause |
| c-ares | 1.34.8-0 | MIT |
| openssl | 3.5.9-0 | Apache-2.0 |
| zlib | 1.3.2-0 | Zlib |

`live-services-host.packman.xml` lists the matching code generators, `bin/protoc` and `bin/grpc_cpp_plugin`, from protobuf and gRPC packages of the same version prefixes. Generator packages are always `static-release`, including for debug target builds, and only their host tools are used.

## Rendering the manifests

Both manifests are templates. The target manifest needs `live_services_target_platform_abi` and `config` (`debug` or `release`); the host manifest needs `live_services_host_platform_abi`, the ABI of the machine that runs the generators. `tools/live/dependencies.py` substitutes both ABIs: `python3 -m tools.live.dependencies --target-abi <target-abi> --host-abi <host-abi>`, run from `client`, writes `deps/live-services-{target,host}.generated.packman.xml` and `deps/live-services-build.json` and leaves `config` for packman to expand. The ABI must match the package producer's naming and agree with the selected USD, compiler and standard library. The `linux-x86_64` platform passes `manylinux_2_35_x86_64` for both, which selects `<version>-manylinux_2_35_x86_64-static-${config}` target packages and `<version>-manylinux_2_35_x86_64-static-release` generators; the `linux-aarch64` platform passes `manylinux_2_35_aarch64` for both. These pins record NVIDIA qualification inputs. Supply matching dependency trees yourself; access to NVIDIA packages is separate from this source release.

Packman links the resolved packages into the `_build` directory of the client, `client/_build` in the repository, which holds links to packman's package cache, not build output; builds and test output go under `../.cache`. The links map to the build inputs of `cmake/LiveGrpcDependencies.cmake`:

- `LIVE_DEPENDENCY_ROOT=<repo>/client/_build/target-deps`
- `LIVE_DEPENDENCY_CONFIG=debug` or `release`
- `LIVE_PROTOC=<repo>/client/_build/host-deps/live-services/protobuf-release/bin/protoc`
- `LIVE_GRPC_CPP_PLUGIN=<repo>/client/_build/host-deps/live-services/grpc-release/bin/grpc_cpp_plugin`

A checkout provisions one target ABI and one host ABI at a time; use separate checkouts for other ABIs. Pass both generator paths explicitly; do not use target executables as generators in a cross build.

## Provisioning

Supply dependencies using the explicit SDK, library and generator paths in the [build guide](../docs/ovlive-build.md). The Packman bootstrap and NVIDIA package acquisition tool are outside this source distribution. The SDK manifests retain the exact qualification pins for provenance.

## Host SDK identities

The native build also consumes these SDKs, which the SDK manifests above pin. `tools/live/build.py` records each selected tree's content identity in the package manifest.

| Input | Identity | Profile |
| --- | --- | --- |
| USD (stock) | OpenUSD 25.11, `pxr` namespace, split `libusd_*.so`, Python-enabled; packman `usd.py312.<abi>.stock.release` `0.25.11-gl.18041+v25.11.363a7c8d` | `stock-25.11` |
| USD (Kit) | `usd.py312.manylinux_2_35_x86_64.ov-full-monolithic.release` `0.25.11.kit.5-gl.21209` (`ov` namespace, `libov_25.11usd_ms.so`); `usd.py312.manylinux_2_35_aarch64.ov-full-monolithic.release` of the same version on `linux-aarch64` | `kit-25.11` |
| Python | 3.12 from the selected USD build: stock profile `python` `3.12.12+nv2-linux-x86_64` or `3.12.12+nv2-linux-aarch64`; Kit profile `python` `3.12.13+nv3-manylinux_2_35_x86_64` or `3.12.13+nv3-manylinux_2_35_aarch64` | both |
| pybind11 | 2.11.1 headers, selected with `LIVE_PYBIND11_ROOT` | both |
| FlatBuffers | 1.12.0 runtime headers, vendored in `source/live/wire/third_party/flatbuffers` | not an input |
| gRPC closure | the manifests above | both |

Carbonite and Omnitrace are not build inputs: no compiled source includes their headers and no manifest lists them. A running Kit host supplies its own Carbonite; the plugin does not link it.

## Test dependencies

The native property and fuzz tests, the seeded cases of the native component drivers such as `live_uri_sanitize`, the envelope, value and scene drivers and the layer progress histories, link no property-testing or fuzzing framework. Every input they draw comes from a pinned source: generated values come from the raw output of `std::mt19937_64`, which the C++ standard fixes for a given seed, and the SDKs they include come from the manifests and host SDK identities above (USD headers from the selected USD build, `grpcpp` and `openssl` from the target manifest, FlatBuffers from its vendored copy). They never use a standard distribution, `std::shuffle` or `std::random_device`, whose results the standard library implementation chooses. `tests/harness/test_suite_catalog.py` enforces both rules, so a new property or fuzz framework is added to a manifest here, with its notice, before a test includes it.
