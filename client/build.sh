#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build with the installed worker configuration; tools/live/local.py describes the inputs.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${OVLIVE_PYTHON:-python3}" -B tools/live/local.py build
