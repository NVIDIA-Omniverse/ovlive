<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ovlive client

The ovlive client is one C++ runtime in the shared library `omni_usd_live` that serves the USD `.live` file format and the `live://` resolver over a Live Services gRPC endpoint, with a C ABI, a C++ USD host adapter, the `omni.usd.live` Python package and a Kit extension. The [ovlive README](../README.md) introduces the product, its quick start, system requirements and licensing; this page maps the client's components.

## Components

| Component | Source and entry points | Responsibility |
|---|---|---|
| **Native interfaces** | [C ABI](include/omni/usd/live/Live.h) · [C++ USD adapter](include/omni/usd/live/LiveUsd.h) · [Reference](docs/ovlive-reference.md) | Runtime and endpoint lifecycle, sessions, status, diagnostics and USD host integration. |
| **Live runtime** | [Runtime source](source/live/runtime/) · [Design](docs/ovlive-design.md) | gRPC transport, credentials, requests, per-layer coordination and recovery. |
| **USD integration** | [Scene source](source/live/scene/) · [Resolver source](source/live/resolver/) | The `.live` file format, scene storage and reconciliation, and `live://` resolution. |
| **Python package** | [Bindings and package](source/bindings/python/omni/usd/live/) · [Python guide](docs/ovlive-python.md) | The `omni.usd.live` API and host helpers. |
| **Kit extension** | [Extension source](source/kit/) · [Kit guide](docs/ovlive-kit.md) | Extension ID `omni_usd_live`, bootstrapped by `omni.kit.live_services`, with public Python API `omni.usd.live`. |
| **Protocol and encoding** | [LiveService protobuf](source/live/protos/nvidia/omniverse/live/v1alpha/live.proto) · [Protocol provenance](source/live/protos/PROVENANCE.md) · [Wire source](source/live/wire/) | The vendored service contract and the OMPF/FlatBuffers USD delta encoding. |
| **Samples** | [Native samples](source/live/samples/) · [Python participant](source/bindings/python/omni/usd/live/sample.py) · [Samples guide](docs/ovlive-samples.md) | Authoring, session, recovery and diagnostics examples. |
| **Tests** | [Native drivers](tests/native/) · [Reference service](tests/reference_service/) · [Testing guide](docs/ovlive-testing.md) | Component, transport, many-writer, package and host-integration validation. The reference service is a test fixture, not a deployable server. |
| **Build and release** | [Build tools](tools/live/) · [Release tools](tools/release/) · [Dependency manifests](deps/README.md) | Provisioned builds, package staging, repository checks and release qualification. |

## Quick start

The steps run from the repository root on Linux. On a machine with an installed [worker configuration](docs/ovlive-ci.md#worker-configuration), `client/build.sh`, `client/test.sh` and `client/clean.sh` build, test and clean without parameters ([local commands](docs/ovlive-build.md#local-commands)). macOS arm64 builds and runs as an experiment, but this platform is unsupported; to try it, follow the [macOS source-build workflow](docs/ovlive-build.md#macos-source-sdks) instead.

**1. Build and stage a package.** Provision the SDKs and tools in the [build prerequisites](docs/ovlive-build.md#prerequisites), then configure, build and stage the package in one step:

```sh
python3 client/tools/live/build.py \
  --platform linux-x86_64 --config release \
  --usd-profile stock-25.11 \
  --usd-root "$LIVE_USD_ROOT" --python-root "$LIVE_USD_PYTHON_ROOT" \
  --pybind11-root "$LIVE_PYBIND11_ROOT" \
  --dependency-root "$LIVE_DEPENDENCY_ROOT" \
  --cmake "$LIVE_CMAKE" --make "$LIVE_MAKE" \
  --cc "$LIVE_CC" --cxx "$LIVE_CXX" \
  --protoc "$LIVE_PROTOC" --grpc-cpp-plugin "$LIVE_GRPC_CPP_PLUGIN" \
  --target-dir "$PWD/.cache/live-package" --parallel 2
```

**2. Check that the package loads.** Set the library and Python paths as [using the package](docs/ovlive-build.md#using-the-package) describes, then run the packaged diagnostics sample. It contacts no service:

```sh
.cache/live-package/usd/omniverse/live/bin/live_diagnostics_sample --help
```

**3. Watch participants exchange edits without a deployed service.** The `plugin-reference` test suite starts the repository's independent reference service on loopback, runs the native participant and the read-only diagnostics sample against it, and checks that the participant's edits arrive. It needs the `live_plugin_driver` test binary, the reference Python 3.12 interpreter and the pinned OpenSSL executable that the [testing guide](docs/ovlive-testing.md#local-setup) describes. To build the driver, add `--build-dir <test-build-dir> --dry-run` to the step 1 command, which prints the build plan as JSON; its first command is the CMake configure command. Run that command with `-DBUILD_TESTING=ON` in place of `-DBUILD_TESTING=OFF`, then `cmake --build <test-build-dir> --target live_plugin_driver`. Reference-service suites run on Linux and Windows, not macOS:

```sh
./client/tools/live/test.sh --suite plugin-reference --seed 42 \
  --reference-python <reference-python> --reference-openssl <openssl> \
  --plugin-driver <test-build-dir>/live_plugin_driver
```

**4. Connect to your own endpoint.** With a Live Services endpoint and an access token it accepts, follow the [Python quick start](docs/ovlive-python.md#quick-start) or run the [participant samples](docs/ovlive-samples.md#participant-samples).

## Working in the client

- **Build and test:** the [build guide](docs/ovlive-build.md) and [testing guide](docs/ovlive-testing.md) give every command and option. Builds and evidence go under the repository root's `.cache`.
- **Documentation:** the [documentation index](docs/README.md) lists every client guide.
- **Changes:** the [changelog](CHANGELOG.md) records client changes.
