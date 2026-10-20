#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provision source-built stock USD and transport SDKs for macOS arm64."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live import platforms, provision_macos_usd, provision_macos_transport


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cmake', default=shutil.which('cmake'))
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--reuse-only', action='store_true', help='validate cached SDKs without building')
    args = parser.parse_args(argv)
    if platforms.host().name != 'macos-arm64':
        parser.error('requires a native macOS arm64 host')
    cmake_path = shutil.which(args.cmake) if args.cmake else None
    if not cmake_path:
        parser.error('install CMake or pass --cmake')
    if not 1 <= args.parallel <= 64:
        parser.error('--parallel must be between 1 and 64')
    cmake_path = str(Path(cmake_path).absolute())
    tools = dict(cmake=cmake_path)
    for name in ('make', 'cc', 'cxx'):
        found = shutil.which('c++' if name == 'cxx' else name)
        if not found:
            parser.error(f'missing build tool: {name}')
        tools[name] = found
    python_root = args.python_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'worker.json'
    # A dependency selection can change as soon as provisioning starts.
    path.unlink(missing_ok=True)
    sdk = ROOT.parent / '.cache/mac-sdk'
    transport = ROOT.parent / '.cache/mac-transport'
    provision_macos_usd.main(['--root', str(sdk), '--python-root', str(python_root),
                             '--cmake', cmake_path, '--parallel', str(args.parallel),
                             *(['--reuse-only'] if args.reuse_only else [])])
    reuse = dict(reuse_only=True) if args.reuse_only else {}
    provision_macos_transport.provision(transport, cmake_path, args.parallel, 'arm64', **reuse)
    pybind = ROOT.parent / '.cache/mac-pybind11'
    if args.reuse_only and not (pybind / '.git').is_dir():
        raise ValueError('Cached pybind11 source checkout is missing')
    provision_macos_usd.checkout(pybind, 'https://github.com/pybind/pybind11.git',
                                '8a099e44b3d5f85b20f05828d919d2332a8de841', **reuse)
    tools.update(protoc=str(transport / 'protobuf-release/bin/protoc'),
                 grpc_cpp_plugin=str(transport / 'grpc-release/bin/grpc_cpp_plugin'))
    config = dict(schema_version=1, purpose='macos-source-build', usd_profile='stock-25.11', tools=tools,
                  sdks=dict(usd_root=str(sdk / 'install'), python_root=str(python_root),
                            dependency_root=str(transport), pybind11_root=str(pybind)))
    with tempfile.TemporaryDirectory(prefix='.worker-', dir=args.output) as directory:
        pending = Path(directory) / 'worker.json'
        pending.write_text(json.dumps(config, indent=2) + '\n')
        pending.replace(path)
    print(path)


if __name__ == '__main__':
    main()
