<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ovlive documentation

This index lists every ovlive document by what a reader wants to do. New readers start with the [quick start](../../README.md#quick-start) in the repository README. See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the contribution policy.

## Use

- [Reference](ovlive-reference.md): semantics, result codes, options, defaults and limits of the C ABI, the C++ host API and the USD integration.
- [Python guide](ovlive-python.md): writing a Python host or tool with `omni.usd.live`, ending with a table of every public name.
- [Kit guide](ovlive-kit.md): loading the `omni.kit.live_services` extension in a Kit application and authoring live layers from Kit.
- [Samples](ovlive-samples.md): running the Python and native participants and the diagnostics sample.

## Build and test

- [Build guide](ovlive-build.md): prerequisites, the build command, the package layout and using the staged package.
- [Testing guide](ovlive-testing.md): the test runner and suites, the reference service, the determinism rules and the qualification gates.
- [CI guide](ovlive-ci.md): the offline CI entry point, the provisioned worker it runs on and the release qualification campaign.
- [Benchmarks](ovlive-benchmarks.md): characterizing the Sdf scene storage and comparing two revisions.

## Design

- [Design](ovlive-design.md): the behavior a host and a service observe, and the reasons for it.

## Operate

- [Diagnostic snapshot schema](ovlive-diagnostic-schema.md): every member and metric of a diagnostic snapshot, and how to validate one.
- [Diagnostic event catalog](ovlive-diagnostic-catalog.md): each diagnostic event, its fields, the diagnostic defaults and the recovery actions.

## Support

- [Support boundary](ovlive-support-boundary.md): the supported configuration, qualification gates, known limitations and open items.

## Release and compliance

- [Source distribution](ovlive-release.md): source scope and build prerequisites.
- [Third-party notices](../THIRD_PARTY_NOTICES.md): client license and dependency evidence.
- [Repository notices](../../THIRD_PARTY_NOTICES.md): client and service license indexes.
