# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The supported build platforms: the host each one builds on and the binary format of its artifacts.

A build is native: the platform it names is the platform of the machine that runs it, never a cross target.
"""
from __future__ import annotations

from dataclasses import dataclass
import platform as _host
import sys


@dataclass(frozen=True)
class Platform:
    name: str
    # platform.system() and the platform.machine() values of a host of this platform.
    system: str
    machines: tuple
    # The executable format of this platform's native artifacts, `elf`, `pe` or `macho`, and the machine field of their headers.
    binary_format: str
    machine_code: int
    # The file name suffix of a CPython 3.12 extension module built for this platform.
    python_extension_suffix: str
    # The dynamic loader a native binary may name as NEEDED, or None where the binary format has no such entry.
    dynamic_loader: str | None
    # The packman platform token of this platform's prebuilt dependencies and SDKs (`platform_target_abi` in Kit's
    # manifests), which also names the platform in a release package version.
    dependency_abi: str


PLATFORMS = {entry.name: entry for entry in (
    Platform('linux-x86_64', 'Linux', ('x86_64', 'AMD64'), 'elf', 0x3E, '.cpython-312-x86_64-linux-gnu.so',
             'ld-linux-x86-64.so.2', 'manylinux_2_35_x86_64'),
    Platform('linux-aarch64', 'Linux', ('aarch64', 'arm64'), 'elf', 0xB7, '.cpython-312-aarch64-linux-gnu.so',
             'ld-linux-aarch64.so.1', 'manylinux_2_35_aarch64'),
    Platform('macos-arm64', 'Darwin', ('arm64',), 'macho', 0x0100000C, '.cpython-312-darwin.so', None,
             'macos-aarch64'),
    Platform('windows-x86_64', 'Windows', ('AMD64', 'x86_64'), 'pe', 0x8664, '.cp312-win_amd64.pyd', None,
             'windows-x86_64'),
)}


def supported(name):
    """Whether `name`, a value of any type, names a supported platform."""
    return isinstance(name, str) and name in PLATFORMS


def get(name):
    """The platform named `name`; an unsupported name is an error."""
    try:
        return PLATFORMS[name]
    except (KeyError, TypeError):
        raise ValueError(f'Unsupported platform: {name}; supported: {", ".join(PLATFORMS)}') from None


def host():
    """The platform of this machine; a machine of no supported platform is an error."""
    system, machine = _host.system(), _host.machine()
    for entry in PLATFORMS.values():
        if entry.system == system and machine in entry.machines:
            return entry
    raise ValueError(f'Unsupported host: {system} {machine}; supported: {", ".join(PLATFORMS)}')


def main(argv=None):
    """`--require <platform>` exits 0 only on a host of that platform, so a worker cannot run another platform's job."""
    import argparse
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--require', required=True, metavar='PLATFORM')
    required = parser.parse_args(argv).require
    try:
        actual = host().name
    except ValueError as error:
        print(f'platform: {error}', file=sys.stderr)
        return 1
    if actual != required:
        print(f'platform: this host is {actual}, not the required {required}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
