<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Source distribution

## Artifact contract

This repository distributes client and service source under the licenses recorded in the [root notice index](../../THIRD_PARTY_NOTICES.md). It includes no client binaries, wheels or Packman bootstrap. The service's [source inventory](../../service/oss/SOURCE-INVENTORY.md) describes its source and separately published service artifacts.

Build the client with the [build guide](ovlive-build.md) and run the service using its [README](../../service/README.md). Supply compatible SDKs and dependencies yourself. The dependency manifests record the exact SDK identities used for NVIDIA qualification; they do not grant public access to NVIDIA package storage. The macOS source SDK scripts fetch from their named public upstream projects. A separately built SDK still needs ABI and host validation.

## Provision release controls

Qualification of this source projection is separate from native package qualification. The [support boundary](ovlive-support-boundary.md) describes supported profiles and outstanding integration gates. Repository automation, release credentials and internal governance checks are outside this source distribution.
