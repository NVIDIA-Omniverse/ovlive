<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Build and package

This guide builds ovlive from source: the shared plugin (`libomni_usd_live.so` on Linux, `libomni_usd_live.dylib` on macOS or `omni_usd_live.dll` on Windows), its public headers, the Python package `omni.usd.live`, the native samples and the staged package that the [Python guide](ovlive-python.md) consumes. The [Kit guide](ovlive-kit.md) consumes `kit-25.11` packages on Linux and Windows; the macOS recipe supplies a standalone `stock-25.11` development package. It is for engineers and agents who build, package or relocate ovlive; the [testing guide](ovlive-testing.md) covers the test suites.

Commands in this guide run from the repository's `client` directory, and the paths they name are relative to it; build output and evidence go to `../.cache`, the `.cache` directory at the repository root.

## Supported profiles

The build targets USD 25.11 and Python 3.12 on the platforms below, selected with `--platform`. A build is native: `build.py` rejects a platform other than the one it runs on, and nothing cross-compiles.

| Platform | Host | Dependency ABI | Python extension suffix |
| --- | --- | --- | --- |
| `linux-x86_64` | Linux x86-64, glibc 2.35 or later | `manylinux_2_35_x86_64` | `.cpython-312-x86_64-linux-gnu.so` |
| `linux-aarch64` | Linux aarch64, glibc 2.35 or later | `manylinux_2_35_aarch64` | `.cpython-312-aarch64-linux-gnu.so` |
| `windows-x86_64` | Windows x86-64 | `windows-x86_64` | `.cp312-win_amd64.pyd` |
| `macos-arm64` | macOS Apple Silicon, native source build | `macos-aarch64` | `.cpython-312-darwin.so` |

Release targets are `kit-25.11` on `linux-x86_64`, `linux-aarch64` and `windows-x86_64`, and `stock-25.11` on `linux-x86_64` ([release procedure](ovlive-release.md)). The macOS source-build path supplies only `stock-25.11` as a development target and has no release qualification lane or Kit support. A platform's package layout does not qualify other profile/platform combinations.

USD ABIs are selected with `--usd-profile` (or `-DLIVE_USD_PROFILE` for direct CMake); macOS provisioning supplies only `stock-25.11`:

| Profile | USD build | Namespace and libraries |
| --- | --- | --- |
| `stock-25.11` (default) | OpenUSD 25.11 with Python enabled | `pxr`, split USD libraries; Linux names are `libusd_*.so`, macOS names are `libusd_*.dylib` |
| `kit-25.11` | Kit's USD 25.11 package | `ov`, monolithic USD; Linux uses `libov_25.11usd_ms.so`, Windows links `ov_25.11usd_ms.lib` and loads the SDK's DLL |

The build checks the USD version, namespace and libraries and records the profile in the package manifest. Every consumer (plugin, resolver, samples, Python extension, tests) links the same selected USD. Other platforms and USD versions have no build profile and fail explicitly. The C ABI in `Live.h` is profile-independent, but the C++ adapter `LiveUsd.h` and the Python USD converters require the exact USD, compiler and standard-library ABI they were built with, so keep stock and Kit builds in separate build and package directories.

## Prerequisites

### macOS source SDKs

The macOS source workflow builds the exact OpenUSD 25.11 source revision and oneTBB revision pinned in [provision_macos_usd.py](../tools/live/provision_macos_usd.py), the transport versions pinned in [provision_macos_transport.py](../tools/live/provision_macos_transport.py), and pybind11 2.11.1. It requires a Python 3.12 framework SDK, CMake, Git, Make, Perl and the Xcode command-line tools. It creates split USD dylibs with Python support and the stock namespace; every native USD consumer must use this same SDK. A separately installed `usd-core` wheel is not part of this environment.

```sh
LIVE_USD_PYTHON_ROOT="$(python3.12 -c 'import sys; print(sys.base_prefix)')"
python3.12 -m venv ../.cache/mac-venv
source ../.cache/mac-venv/bin/activate
python -m pip install cmake==3.31.6
python -B tools/live/provision_macos.py --python-root "$LIVE_USD_PYTHON_ROOT" --output ../.cache/mac-worker
```

The resulting `../.cache/mac-worker/worker.json`, marked `purpose: macos-source-build`, names the source-built SDK roots and tools. The `macos_ci.py` command below fixes the platform to `macos-arm64` and profile to `stock-25.11`; it takes neither a `--platform` nor a `--usd-profile` option. Provisioning removes this configuration before changing dependencies and publishes it atomically after every step succeeds; a failed provision leaves no usable configuration. Build artifacts and source checkouts stay under `../.cache`. The USD/oneTBB build provenance records source commits, configure/build/install commands, toolchain identity and installed-file hashes; transport completion markers record source pins, recipe/toolchain identity and installed-file hashes, and pybind11 is a checkout validated against its pinned commit. Transport commands appear in the provisioning output, without a separate command record. Set `DYLD_LIBRARY_PATH` to the selected USD and Python SDK `lib` directories when loading native artifacts. The plugin uses `lib/libomni_usd_live.dylib`, the Python extension uses the Darwin suffix, and package-relative library paths use `@loader_path`. Mach-O validation checks arm64 architecture and the package retains the same single plugin and public C ABI.

For the hosted package testing, staging and archive checks locally, use a fresh output directory and a clean committed checkout:

```sh
python -B tools/live/macos_ci.py --config ../.cache/mac-worker/worker.json --output ../.cache/mac-ci --parallel 4
python -B tools/release/archive.py create --package ../.cache/mac-ci/stage/usd/omniverse/live \
  --output-dir ../.cache/mac-ci/archives --extract-to ../.cache/mac-ci/archive-check
```

The `macos_ci.py` entry point reads the validated configuration, checks the exact ten-test inventory and results, and checks staged native/Python load orders. The separate `archive.py create` command creates the development archive and checksum, checks them and validates the extracted layout; neither command uploads artifacts. To run only the six package/sample CTests directly, read the selected inputs from that same configuration and configure a separate test tree with `BUILD_TESTING=ON`. This subset omits the four component CTests, staged relocation checks and archive checks:

```sh
mac_input() {
  python -B -c 'import sys; from tools.live.macos_ci import read_config; print(read_config("../.cache/mac-worker/worker.json")[sys.argv[1]][sys.argv[2]])' "$@"
}
LIVE_CMAKE="$(mac_input tools cmake)"
LIVE_MAKE="$(mac_input tools make)"
LIVE_CC="$(mac_input tools cc)"
LIVE_CXX="$(mac_input tools cxx)"
LIVE_PROTOC="$(mac_input tools protoc)"
LIVE_GRPC_CPP_PLUGIN="$(mac_input tools grpc_cpp_plugin)"
LIVE_USD_ROOT="$(mac_input sdks usd_root)"
LIVE_USD_PYTHON_ROOT="$(mac_input sdks python_root)"
LIVE_PYBIND11_ROOT="$(mac_input sdks pybind11_root)"
LIVE_DEPENDENCY_ROOT="$(mac_input sdks dependency_root)"
LIVE_CTEST="$(dirname "$LIVE_CMAKE")/ctest"
```

```sh
"$LIVE_CMAKE" -S . -B ../.cache/mac-tests -G "Unix Makefiles" \
  -DCMAKE_MAKE_PROGRAM="$LIVE_MAKE" \
  -DCMAKE_C_COMPILER="$LIVE_CC" -DCMAKE_CXX_COMPILER="$LIVE_CXX" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF \
  -DLIVE_USD_PROFILE=stock-25.11 \
  -DLIVE_USD_ROOT="$LIVE_USD_ROOT" \
  -DLIVE_USD_PYTHON_ROOT="$LIVE_USD_PYTHON_ROOT" \
  -DLIVE_PYBIND11_ROOT="$LIVE_PYBIND11_ROOT" \
  -DLIVE_DEPENDENCY_ROOT="$LIVE_DEPENDENCY_ROOT" \
  -DLIVE_DEPENDENCY_CONFIG=release \
  -DLIVE_PROTOC="$LIVE_PROTOC" -DLIVE_GRPC_CPP_PLUGIN="$LIVE_GRPC_CPP_PLUGIN" \
  -DLIVE_PYTHON_EXECUTABLE="$LIVE_USD_PYTHON_ROOT/bin/python3.12" \
  -DPython3_EXECUTABLE="$LIVE_USD_PYTHON_ROOT/bin/python3.12" \
  -DLIVE_BUILD_WIRE=ON -DLIVE_BUILD_SCENE_CORE=ON -DLIVE_BUILD_USD_WIRE=ON \
  -DLIVE_BUILD_WIRE_SCENE=ON -DLIVE_BUILD_GRPC_RUNTIME=ON \
  -DLIVE_BUILD_SCENE_TRANSPORT=ON -DLIVE_BUILD_PLUGIN=ON -DLIVE_BUILD_PYTHON=ON
"$LIVE_CMAKE" --build ../.cache/mac-tests --parallel 4
DYLD_LIBRARY_PATH="$LIVE_USD_ROOT/lib:$LIVE_USD_PYTHON_ROOT/lib" \
  "$LIVE_CTEST" --test-dir ../.cache/mac-tests --no-tests=error --output-on-failure \
  -R '^live_(participant_tests|diagnostic_log_tests|diagnostic_log_json|plugin_package|python_package|installed_package)$'
```

Run the macOS source SDK and package commands above on a native arm64 host.

The build consumes already-provisioned SDKs and tools and never fetches anything. Supply each as an absolute path:

| Input | Requirement | `build.py` flag |
| --- | --- | --- |
| USD | USD 25.11 of the selected profile, Python-enabled | `--usd-root` |
| Python | CPython 3.12 matching the USD build: Linux `<python-root>/include/python3.12/`, `<python-root>/lib/libpython3.12.so`, `<python-root>/bin/python3.12`; macOS framework SDK `<python-root>/include/python3.12/`, `<python-root>/lib/libpython3.12.dylib`, `<python-root>/bin/python3.12`; Windows `<python-root>/include/`, `<python-root>/libs/python312.lib`, `<python-root>/python.exe` | `--python-root` |
| pybind11 | 2.11.1 headers | `--pybind11-root` |
| Transport libraries | gRPC 1.65.4, protobuf 5.26.0, abseil, re2, c-ares, openssl and zlib, laid out as `<package>-<config>/{include,lib}` | `--dependency-root` |
| Code generators | Host `protoc` 26.0 and `grpc_cpp_plugin` matching the transport libraries | `--protoc`, `--grpc-cpp-plugin` |
| CMake | 3.20 or later | `--cmake` |
| Compilers | C and C++17 compilers | `--cc`, `--cxx` |
| Build tool | Make or Ninja | `--make` or `--ninja` |

The FlatBuffers runtime headers are vendored in `source/live/wire/third_party/flatbuffers`. The [dependency manifests](../deps/README.md) record the exact transport versions and the [host SDK identities](../deps/README.md#host-sdk-identities) for each profile. Where packman is available, render the manifests for explicit target and host ABIs and provision them into the dependency root:

```sh
python3 -m tools.live.dependencies --target-abi manylinux_2_35_x86_64 --host-abi manylinux_2_35_x86_64
python3 -m tools.live.dependencies --target-abi manylinux_2_35_aarch64 --host-abi manylinux_2_35_aarch64
```

Run the matching manifest-rendering command from `client`. Rendering records dependency identities; supply the dependency trees and code generators through the explicit paths below. Package download automation is outside this source distribution.

Without packman, build gRPC 1.65.4, protobuf 5.26.0, abseil 20240116.1, re2 20230301, c-ares 1.34.8, OpenSSL 3.5.9 and zlib 1.3.2 as static libraries, plus the host `protoc` and `grpc_cpp_plugin`. Lay each library out as `<dependency-root>/<package>-<config>/` holding `include/`, `lib/` and `PACKAGE-LICENSES/`, where `<package>` is `grpc`, `protobuf`, `abseil`, `re2`, `c-ares`, `openssl` or `zlib` and `<config>` is `debug` or `release`. [`cmake/LiveGrpcDependencies.cmake`](../cmake/LiveGrpcDependencies.cmake) lists the static archives it links from each `lib/`; a debug protobuf archive is named `protobufd`. The CI `notices` step reads each license from `PACKAGE-LICENSES/`: `grpc-LICENSE.txt`, `protobuf-LICENSE.txt`, `abseil-LICENSE.txt`, `re2-LICENSE.txt`, `c-ares-LICENSE.md`, `openssl-LICENSE.txt` and `zlib-LICENSE.txt`. It also requires each `<package>-<config>` path to resolve to a directory named after the manifest version up to the ABI tag (`1.65.4-1` for gRPC, `5.26.0-2` for protobuf; a directory named `1.65.4` fails), optionally followed by `-` and a suffix; the `version` attributes in [`deps/live-services-target.packman.xml`](../deps/live-services-target.packman.xml) give each one.

## Build

`tools/live/build.py` configures, builds, installs and stages the package in one step. This shell example selects the Linux x86-64 stock target; the Windows release target selects `--platform windows-x86_64 --usd-profile kit-25.11` with Windows SDKs and tools, and the macOS recipe above selects its fixed development target:

```sh
python3 tools/live/build.py \
  --platform linux-x86_64 --config release \
  --usd-profile stock-25.11 \
  --usd-root "$LIVE_USD_ROOT" --python-root "$LIVE_USD_PYTHON_ROOT" \
  --pybind11-root "$LIVE_PYBIND11_ROOT" \
  --dependency-root "$LIVE_DEPENDENCY_ROOT" \
  --cmake "$LIVE_CMAKE" --make "$LIVE_MAKE" \
  --cc "$LIVE_CC" --cxx "$LIVE_CXX" \
  --protoc "$LIVE_PROTOC" --grpc-cpp-plugin "$LIVE_GRPC_CPP_PLUGIN" \
  --target-dir "$PWD/../.cache/live-package" --parallel 2
```

| Flag | Meaning |
| --- | --- |
| `--config` | `debug` or `release` |
| `--make` / `--ninja` | Exactly one build tool; selects the CMake generator |
| `--build-dir` | CMake cache directory; default `../.cache/live-repo/<platform>/<config>` |
| `--parallel` | Build parallelism, 1 to 64; default 2 |
| `--dry-run` | Validate the inputs and print the commands without building or publishing |

A build cache rejects a changed tool or SDK selection instead of reusing a different compiler or ABI; use a fresh `--build-dir` to switch. The production build sets `BUILD_TESTING=OFF` and builds the Python extension with `<python-root>/bin/python3.12` on Linux/macOS or `<python-root>/python.exe` on Windows, so the interpreter that runs `build.py` does not choose the extension ABI.

### Direct CMake

CMake is the only build system; `build.py` is a wrapper that passes these cache variables. A direct build sets `LIVE_USD_PROFILE`, `LIVE_USD_ROOT`, `LIVE_USD_PYTHON_ROOT`, `LIVE_PYBIND11_ROOT`, `LIVE_DEPENDENCY_ROOT`, `LIVE_DEPENDENCY_CONFIG` (`debug` or `release`), `LIVE_PROTOC` and `LIVE_GRPC_CPP_PLUGIN`, and enables the components with `LIVE_BUILD_SCENE_CORE`, `LIVE_BUILD_WIRE`, `LIVE_BUILD_USD_WIRE`, `LIVE_BUILD_WIRE_SCENE`, `LIVE_BUILD_GRPC_RUNTIME`, `LIVE_BUILD_SCENE_TRANSPORT`, `LIVE_BUILD_PLUGIN` and `LIVE_BUILD_PYTHON`. CMake rejects a `protoc` that does not report version 26.0. `LIVE_BUILD_SCENE_CORE` alone builds the exact Sdf scene storage and the support files in `source/live/support` against the selected USD and Python 3.12, which is enough for the [storage benchmarks](ovlive-benchmarks.md).

## Local commands

`./build.sh`, `./test.sh` and `./clean.sh` in `client` take no parameters. They run `tools/live/local.py`, which reads the [worker configuration](ovlive-ci.md#worker-configuration) that names every SDK and tool, from `OVLIVE_WORKER_CONFIG` or `$XDG_CONFIG_HOME/ovlive/worker.json` (`$HOME/.config/ovlive/worker.json` when `XDG_CONFIG_HOME` is unset). `OVLIVE_PYTHON` selects the interpreter that runs them (default `python3`).

| Command | What it does |
| --- | --- |
| `./build.sh` | The `presubmit` profile's configure and build steps into `../.cache/local/build`, with test drivers and the Python extension; a second run reuses the CMake cache |
| `./test.sh` | The `presubmit` profile's contract suites against that build, with seed 42 and artifacts under `../.cache/local/tests`; it needs a completed `./build.sh` |
| `./clean.sh` | Removes `../.cache/local` |

The commands download nothing and discover no SDK: a missing configuration is an error that names the expected path. `tools/live/provision.py` writes a configuration for one USD profile ([prerequisites](#prerequisites)); point `OVLIVE_WORKER_CONFIG` at its `worker.json`. They never write to `../.cache/ci`; a qualification run is a fresh `tools/live/ci.py` invocation as the [CI guide](ovlive-ci.md#running) describes.

## Package layout

The package lands at `<target-dir>/usd/omniverse/live`:

| Path | Contents |
| --- | --- |
| `lib/libomni_usd_live.so` (Linux), `lib/libomni_usd_live.dylib` (macOS), `bin/omni_usd_live.dll` (Windows) | The shared plugin: USD file format, resolver and C ABI |
| `resources/plugInfo.json` | USD plugin registration |
| `config/` | `extension.toml` and `live-sdk.json` |
| `include/omni/usd/live/` | `Live.h` (C ABI) and `LiveUsd.h` (C++ USD adapter) |
| `bin/` | `live_diagnostics_sample` and `live_participant_sample` ([samples](ovlive-samples.md)) |
| `python/omni/usd/live/` | The Python package, its extension module, type stub and `sample.py` |
| `python/omni/kit/live_services/` | The Kit bootstrap module |
| `share/omni_usd_live/` | Apache `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, the license bundle `THIRD_PARTY_LICENSES.txt`, the bill of materials `BOM.json` and upstream provenance for the wire schema, protocol and FlatBuffers headers |
| `package-manifest.json` | Hashes binding the package to its source, configuration and SDK and tool identities |

There are 29 payload files plus the manifest on Linux and macOS. On `windows-x86_64` the plugin library is `bin/omni_usd_live.dll`, beside the sample executables `bin/live_diagnostics_sample.exe` and `bin/live_participant_sample.exe` where the Windows loader finds it, the extension module is `_live.cp312-win_amd64.pyd`, which the Python package lets import the DLL by adding `bin` to the DLL search path, and the package also holds the import library `lib/omni_usd_live.lib` for C ABI consumers: There are 30 payload files plus the manifest. Validation rejects missing, modified or extra members, symlinks, stale inputs, a wrong platform or configuration and a sibling resolver registration. Publication replaces only a package directory `build.py` owns and keeps the previous package if building or validation fails; a destination holding an unrecognized package is rejected, so publish into a fresh tree. The client NOTICE, third-party notices, license bundle and BOM are installed from `client/`, and LICENSE from the repository root; the CI `notices` step checks them against the license files of the provisioned dependencies ([CI guide](ovlive-ci.md)).

The package is immutable: validation rejects any extra file, including `__pycache__` directories and `.pyc` files. A CMake install produces the same files under an install prefix. Linux samples search `$ORIGIN/../lib` and then `$ORIGIN/../lib/omni_usd_live/lib`; macOS samples use the corresponding `@loader_path` paths. Windows samples load the plugin DLL beside their executables in `bin`. Each platform finds the plugin in its staged and CMake install layouts.

## Release archive

The release archive is the qualification record of a release target: the [release procedure](ovlive-release.md) qualifies one archive per target and packs the target's packman package, its one binary release asset, from it; the archive itself is not a release asset. The archive is built from a validated `release` stage: `python3 -B tools/release/archive.py create --package <target-dir>/usd/omniverse/live --output-dir <dir>` writes `ovlive-<version>-<usd-profile>-<platform>.tar.gz` and a `.sha256` beside it, where `<platform>` is the stage's platform, which must be the platform of the machine that archives it. The version is the `[package] version` of `source/kit/config/extension.toml`; a `vX.Y.Z` tag on the head must match it.

The archive holds one directory `ovlive-<version>-<usd-profile>-<platform>/` with the package's 29 payload files on Linux/macOS or 30 on Windows, byte-identical to the stage, under `omni_usd_live/`, the Kit extension folder name, and `release-manifest.json`: the version, source revision and fingerprint, platform, USD profile, ABI version, the SHA-256 of the SDK trees and tools, the bill of materials and each member's SHA-256, size and mode. It holds no `package-manifest.json` and names no path of the machine that built it, apart from the build paths under `/builds/omniverse` that the prebuilt dependencies record in their own static libraries. Members carry the head commit's time, or `SOURCE_DATE_EPOCH` when it is set, and two archives of one stage made with one Python and zlib build are byte-identical. The macOS development archive uses this same archive format without becoming a qualified release asset.

For Linux builds, the build maps the checkout, the build tree, the SDK roots and each dependency package's resolved directory out of `__FILE__` strings and debug information with `-ffile-prefix-map` (to `.`, `<build>`, `<usd>`, `<python>`, `<deps>` or `<deps>/<package>-<config>` and `<pybind11>`), so a debugger needs `set substitute-path . <checkout>` and `set substitute-path <build> <build-dir>`. GCC builds with `-fno-canonical-system-headers`, so a dependency header keeps the spelling of the include directory it was found in, which a map names, rather than a shorter symlink target in the package manager's cache. The Linux production `build.py` path links the package binaries with their install RPATH (`CMAKE_BUILD_WITH_INSTALL_RPATH`) and lets the linker find the USD and Python libraries through `-rpath-link`: a build tree RPATH names the build directory, and installation rewrites only the RPATH string, which leaves the build ID and the dynamic string table size of each binary that links the plugin dependent on the build directory. CI and local test builds, including `macos_ci.py`, run their tests from the build tree and keep the build tree RPATH. The binaries ship unstripped (`"stripped": false`): symbol tables and the prebuilt dependencies' debug sections are kept; the product code is compiled without `-g`.

`python3 -B tools/release/archive.py check --archive <archive> [--extract-to <fresh dir>]` verifies the `.sha256` checksum file beside the archive, which must exist, rejects links, devices, absolute or parent-relative members and unexpected modes, re-hashes every member against the manifest, rejects a member that contains this checkout's path or a `--forbid` path as a whole path or path prefix (`create` forbids the stage's checkout, SDK roots and tool roots) and, with `--extract-to`, extracts and validates the exact package layout. `create` checks the archive under a temporary name and publishes it and its checksum only if the check passes; it refuses a debug stage, a stale stage, an embedded revision other than the head and, without `--allow-dirty`, a modified checkout. The path match has two blind spots: it does not report a forbidden path that follows a path character, such as a length-prefixed string whose length byte is a printable character, or that follows `//`. `check --archive <archive> --against <other>` checks both archives, which must have one name, then compares them member by member and prints each member whose content or header differs, each top-level release manifest field other than `files` that differs, and the count of differing members; it exits 0 only when no member differs and the two files are byte-identical. The CI `package-archive` step records `release_archive` in the run summary, the archive name, SHA-256, `git_head` and `dirty`, so two runs of one commit can be paired and their archives compared. The step passes `--allow-dirty` and records the checkout it archived as `checkout` in the run summary: `state` is `clean`, `modified` or, when git or the repository is unavailable, `unknown`, and `changed_files` is the count of modified, staged and untracked non-ignored files, or null for `unknown`.

## Using the package

USD and Python shared libraries are host SDK inputs and are not copied into the package. To import the Python package or run a sample from the staged package, use the matching SDK's Python 3.12. The following environment example is for Linux:

```sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="<target-dir>/usd/omniverse/live/python:<USD-root>/lib/python"
export LD_LIBRARY_PATH="<USD-root>/lib:<python-root>/lib"
```

On macOS, use the same `PYTHONDONTWRITEBYTECODE` and `PYTHONPATH` values and set `DYLD_LIBRARY_PATH` to `<USD-root>/lib:<python-root>/lib`. On Windows, separate the two `PYTHONPATH` directories with `;`, add the USD SDK's DLL directory to `PATH`, and run the matching SDK's `python.exe`; the `omni.usd.live` package adds its own `bin` directory to the DLL search path. Set `PYTHONDONTWRITEBYTECODE=1` on Windows as well.

`PYTHONDONTWRITEBYTECODE=1` keeps the interpreter from writing bytecode into the immutable package; keep other generated caches outside it too. No Kit installation is needed for the standalone package. `bin/live_diagnostics_sample --help` (`bin/live_diagnostics_sample.exe --help` on Windows) runs without an endpoint and confirms that the package loads. The C ABI, the USD plugin and the Python extension share one library for the life of the process; do not unload either native module. The [Python guide](ovlive-python.md) and the [Kit guide](ovlive-kit.md) continue from here.

## Next steps

- [Testing guide](ovlive-testing.md): the test suites and how to run them locally.
- [CI guide](ovlive-ci.md): the offline entry point for a provisioned worker.
- [Support boundary](ovlive-support-boundary.md): what the supported profiles cover.
