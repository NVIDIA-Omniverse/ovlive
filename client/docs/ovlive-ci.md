<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CI guide

`tools/live/ci.py` builds ovlive and runs its full SDK test suite offline on a provisioned worker of one [build platform](ovlive-build.md#supported-profiles), `linux-x86_64` or `linux-aarch64`; it builds for the platform of the machine it runs on. This guide is for engineers who set up that worker or run the entry point by hand; the [testing guide](ovlive-testing.md) describes the suites it runs, and the [support boundary](ovlive-support-boundary.md) states what CI covers.

Commands in this guide run from the repository's `client` directory, and the paths they name are relative to it; build output and evidence go to `../.cache`, the `.cache` directory at the repository root.

## Workflows

Run the build and test entry points below on a host with the required SDKs. Hosted release workflows and their provisioning are outside this source distribution. Tests specific to those excluded tools and internal repository governance are excluded with them; source and runtime tests remain available.

## Profiles

Every profile builds Release binaries with test assertions enabled, runs the mandatory suites (`soak` runs only `writer-fuzz`), installs the package and checks the installed native package.

| Profile | Sanitizers | Python adapter and `python-reference` | Writer campaign | Contract step watchdog |
| --- | --- | --- | --- | --- |
| `presubmit` (default) | none | yes | `ci` | 40 minutes |
| `asan` | ASan and UBSan with leak detection, suppressions in `tools/live/lsan.supp` | no | `ci` | 40 minutes |
| `nightly` | none | yes | `nightly` | 90 minutes |
| `tsan` | ThreadSanitizer with suppressions in `tools/live/tsan.supp`; instrumented steps run under `setarch <machine> -R`, where `<machine>` is `x86_64` or `aarch64`; `stock-25.11` only | no | `ci` | 90 minutes |
| `soak` | none | built, not tested (no `python-reference`) | `soak` (one case) | 210 minutes |

The watchdogs are failure limits sized for the slowest qualified platform, `windows-x86_64`, where the contract suites run three to five times as long as on `linux-x86_64`; no passing run waits on one.

The `tsan` profile runs about 2.7 times as long as `asan` (per-suite watchdog 50 minutes), so it runs beside `nightly`, not on every pull request.

The `soak` profile runs only the `writer-fuzz` suite with one case: 4 plugin writers through 600 backend rotations with a 5-second READY pause each, about 60 minutes; its case watchdog is 200 minutes. It runs weekly and before a release candidate, not beside every `nightly`.

The mandatory suites are `foundation`, `lock-noalloc`, `wire-envelope`, `usd-values`, `wire-scene`, `scene-journal`, `sdf-scene`, `runtime-foundation`, `diagnostics-schema`, `reference-service`, `native-reference`, `scene-reference`, `resolver-reference`, `plugin-reference`, `writer-fuzz` and `resolver-probe`; `presubmit` and `nightly` add `uri` and `python-reference` and verify the installed Python package from outside the checkout. The `kit-reference` suite needs a Kit build and runs separately ([Kit guide](ovlive-kit.md#running-the-kit-reference-lane)).

## Steps

The entry point runs these steps in order, each under a failure watchdog, and stops at the first failure without retrying:

| Step | What it does | Watchdog |
| --- | --- | --- |
| `reference-identity`, `diagnostic-identity` | Checks the package versions of the reference-service and schema interpreters | 30 seconds each |
| `notices` | Runs `tools/release/third_party_notices.py --check` against the provisioned dependency root, so `THIRD_PARTY_LICENSES.txt` and `BOM.json` must match the dependencies' license files | 30 seconds |
| `configure`, `build` | CMake configure and build | 2 and 30 minutes |
| `contracts` | `tools/live/test.py` with the profile's suites, seed and writer campaign | per profile |
| `uri-native` | Native URI bounds executable | 1 minute |
| `stack-dump` | Checks that a native test driver answers SIGQUIT with every thread's stack, as the runner's watchdog uses it | 1 minute |
| `install` and the two package checks | Installs the package, checks that its notices match the repository and (except in the sanitizer profiles) that its layout is exact, then checks native load order after relocation and, where the profile builds Python, the Python package from outside the checkout | 2 minutes each |
| `package-stage`, `package-archive` and the two archive checks | In `presubmit` and `nightly`: builds the release package with `tools/live/build.py` in its own build directory, archives and checks it with `tools/release/archive.py`, then checks native load order after relocation and the Python package on the extracted `omni_usd_live` folder | 30, 5 and 2 minutes |

A step whose watchdog expires captures thread stacks the same way ([testing guide](ovlive-testing.md#determinism-rules)).

Each step keeps at most 16 MiB of output; truncation fails the run.

## Running

```sh
python3 -B tools/live/ci.py --config /worker/ovlive.json \
  --output "$PWD/../.cache/ci-presubmit" --profile presubmit --seed 42 --parallel 2
```

`--output` must be a fresh directory, including after a failed run. `--seed` is a nonnegative signed 64-bit integer; `--parallel` is 1 to 8. `--dry-run` validates the configuration and prints the complete plan without running it. The output directory holds the plan, source fingerprint, runner and SDK identities, step logs, test manifests, JUnit and replay data.

`python3 -B tools/live/summarize.py <output dir>` prints a run's result, profile, seeds, source fingerprint, Git head and per-suite test counts from an output directory or a `tools/live/test.py` artifact directory, and exits nonzero unless every result passed.

## Worker configuration

`--config` names one JSON object with `schema_version: 1`, `tools`, `sdks` and optionally `usd_profile` (`stock-25.11` by default, or `kit-25.11`). Every value is an absolute path to an existing input; select exactly one of `make` or `ninja`. Duplicate, unknown, missing or relative entries and files over 32 KiB are rejected. The file never holds credentials, commands, environment overrides or test selections. The [local commands](ovlive-build.md#local-commands) read the same file from `OVLIVE_WORKER_CONFIG` or `$XDG_CONFIG_HOME/ovlive/worker.json`.

```json
{
  "schema_version": 1,
  "usd_profile": "stock-25.11",
  "tools": {
    "cmake": "/tools/cmake",
    "make": "/tools/make",
    "cc": "/tools/cc",
    "cxx": "/tools/c++",
    "protoc": "/tools/protoc",
    "grpc_cpp_plugin": "/tools/grpc_cpp_plugin",
    "reference_python": "/reference/bin/python",
    "reference_openssl": "/openssl/bin/openssl",
    "diagnostic_python": "/diagnostic/bin/python"
  },
  "sdks": {
    "usd_root": "/sdk/usd",
    "python_root": "/sdk/python",
    "dependency_root": "/sdk/transport",
    "pybind11_root": "/sdk/pybind11"
  }
}
```

The SDK and tool inputs are those of the [build guide](ovlive-build.md#prerequisites). `reference_python` needs the packages `tools/live/reference.py` requires, and `diagnostic_python` the packages `tools/live/diagnostic_snapshot.py` requires.

## Worker setup

1. Use a worker of the selected build platform with the SDKs, tools and configuration described above. The release workflow selects its self-hosted workers by the platform labels in [Provision release controls](ovlive-release.md#provision-release-controls); standalone qualification uses the worker on which `tools/live/ci.py` runs.
2. Install the SDKs, tools and configuration outside the checkout and pass their paths explicitly.
3. On Linux, install a host Python whose `os` and `signal` modules provide `pidfd_open` and `pidfd_send_signal`; the entry point uses them to contain every workload process and fails without them. Windows requires the Job Object containment described in the [testing guide](ovlive-testing.md#service-readiness-and-cleanup).
4. Run the [entry point](#running) with the worker configuration and a fresh output directory for each attempt. Retain its plan, summary, logs and test results, including failures, as qualification evidence. Installing dependencies alone supplies no qualification result.

## Qualification campaign

A release candidate needs the evidence of one qualification campaign before the [support boundary](ovlive-support-boundary.md#qualification-boundary) can claim the gates the campaign covers for that commit. The campaign runs every CI profile plus a few repeated runs that no single profile covers, and records each result against the commit it ran at. Two gates take their evidence elsewhere: Kit integration from the `kit-reference` lane on the Kit profile ([Kit guide](ovlive-kit.md#running-the-kit-reference-lane)) and the manual Kit run the open item `kit-rerun` tracks, and Performance from the [storage benchmarks](ovlive-benchmarks.md), which have no budgets to qualify (`gate-performance`). The campaign needs one worker configuration per USD profile (`usd_profile` set to `stock-25.11` and to `kit-25.11`) and the [reference interpreter](ovlive-testing.md#local-setup).

Run the steps in this order. The cheap checks come first so a shape or harness failure stops the campaign before hours of worker time, and the later steps use the builds the earlier ones produce:

1. Check formatting with `python3 -B tools/release/check_format.py --clang-format <clang-format 18.1.1>` and run the source tests below.
2. The `foundation` suite on the host Python and on the worker configuration's `reference_python`. The campaign covers two Python versions only when that interpreter's version differs from the host Python's, so the record names both versions.
3. `presubmit` on both USD profiles, then CTest on each profile's build directory with the `ctest` beside the worker configuration's `cmake` (`ctest --test-dir` needs CMake 3.20 or later); every CTest test passes, and the `presubmit` builds disable none (the sanitizer builds disable `live_uri_contract`).
4. `asan` on `stock-25.11`.
5. `nightly` on both USD profiles; each run also stages, archives and checks the release archive, and its summary records the archive name, SHA-256, source revision and dirty state; `tools/release/archive.py check --archive <archive> --against <other>` compares the archives of two runs of one commit member by member.
6. The `nightly` writer campaign at two further master seeds chosen for the record, for example 7 and the release date as a number, with the plugin and scene transport drivers of the `nightly` run's build for that USD profile, so the campaign covers more seeded histories of the same binaries.
7. `tsan` on `stock-25.11` only, at two seeds, each run alone, because ThreadSanitizer is timing-sensitive and other load changes which interleavings it sees; `ci.py` refuses the `kit-25.11` configuration for `tsan`.
8. The auth-watchdog probe: the `actor-auth-timeout` native driver mode 50 times against each `presubmit` build's unary driver, each iteration against a fresh reference service; every iteration applies its edit and none reaches the auth terminal watchdog. This probes the open item `test-intermittent`.
9. Two stress runs against the `presubmit` stock build's Python extension: `test_python_host.PythonHostTests.test_python_plain_usd_edits_converge_between_two_hosts` and `test_python_creation.PythonCreationTests.test_python_creation_dispatch_failure_keeps_the_host_layer_exportable`, each as 8 parallel processes of 40 iterations, to surface ordering defects that a single pass hides; every one of the 320 iterations of each test passes.
10. The `soak` profile on `stock-25.11`, alone, because its memory and thread checks compare against its own warm-up values and other load distorts them.

Step 8 runs `./tools/live/test.sh --suite native-reference --driver-mode actor-auth-timeout` with the reference inputs and the build's `--unary-driver`, once per iteration, each run starting a fresh reference service ([selecting suites and cases](ovlive-testing.md#selecting-suites-and-cases)). `tools/live/test.py` has no option that repeats one test, which step 9 needs, and that option has no open item yet; step 9 therefore runs outside the tracked entry points, and its record states the command that ran it.

Run at most two profiles at once, such as the two USD profiles of `presubmit` or of `nightly`. Concurrent load changes timing, so a failure under more load is harder to attribute to the product rather than to the worker.

Plan worker time from these measurements of one run with `--parallel 8` on a 24-core worker: `presubmit` about 5 minutes per USD profile, `asan` about 10, `nightly` about 9 per USD profile, each `tsan` seed about 14, `soak` about 54, each writer-campaign seed about 5, the auth-watchdog probe about 1 per build and the stress pair about 3; a worker with fewer cores takes longer, so plan with about twice these figures until it has its own measurements.

The commands take the shapes of [Running](#running) and the [testing guide](ovlive-testing.md#running-the-tests), each with a fresh output or artifact directory:

```sh
./tools/live/test.sh --suite foundation --seed 42 --artifact-dir <fresh dir>
LIVE_TEST_PYTHON=<reference python> ./tools/live/test.sh --suite foundation --seed 42 --artifact-dir <fresh dir>
python3 -B tools/live/ci.py --config <worker config> --output <fresh dir> --profile presubmit --seed 42 --parallel 8
ctest --test-dir <presubmit output>/build  # the ctest beside the configured cmake
python3 -B tools/live/test.py --suite writer-fuzz --writer-fuzz-profile nightly --seed <seed> \
  --plugin-driver <nightly output>/build/live_plugin_driver \
  --scene-transport-driver <nightly output>/build/live_scene_transport_driver \
  --reference-python <reference python> --reference-openssl <openssl 3.5.9> --artifact-dir <fresh dir>
python3 -B tools/live/ci.py --config <stock worker config> --output <fresh dir> --profile tsan --seed <seed> --parallel 8
```

Record every run as it finishes, one row per run:

| Step | CI profile | USD profile | Seed | Commit | Source fingerprint | Counts | Result | Command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | `presubmit` | `stock-25.11` | 42 | `<commit>` | `<fingerprint>` | `<tests per suite>`; CTest `<passed>/<total>` | passed | `tools/live/ci.py ... --profile presubmit --seed 42` |

`python3 -B tools/live/summarize.py <output dir>` prints every column except Step, the CTest and iteration counts and the command from a CI output or test artifact directory, and exits nonzero unless the run passed. From `client`, `python3 -B -c 'from tools.live.test import fingerprint; print(fingerprint())'` prints the fingerprint of the checkout itself; it matches the fingerprint of every run taken at that checkout. The record lives in the release pull request's description, and the reviewer who merges that pull request signs it off.

A record names the commit each run was taken at. When one record holds runs from more than one commit, it lists every file that differs between those commits (`git diff --name-only <commit> <commit>`), so a reader can judge whether the difference can affect the earlier runs.

A failure is a result, not noise. A passing rerun does not count as evidence and does not replace the failure in the record: find the root cause, fix it, and rerun the affected profiles at the new head, which the record then names.
