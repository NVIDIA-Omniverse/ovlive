#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check or apply the pinned clang-format over NVIDIA-owned C and C++ sources."""
import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CLANG_FORMAT_VERSION = '18.1.1'
# The pinned PyPI wheel appends its build provenance to the upstream banner; distribution builds
# prefix a vendor name or append a package revision and carry vendor patches, so they never match.
WHEEL_PROVENANCE = r'(?: \(https://github\.com/ssciwr/clang-format-wheel [0-9a-f]{7,64}\))?'
DIRECTORIES = ('source/live', 'include/omni/usd/live', 'source/bindings')
# X-macro tables (.inc) keep their hand layout; pinned upstream copies keep their bytes.
SUFFIXES = ('.c', '.cc', '.cpp', '.h', '.hpp', '.inl')
EXCLUDED = ('source/live/wire/schema/', 'source/live/wire/third_party/')


def sources(root=ROOT):
    names = subprocess.check_output(['git', 'ls-files', '-z', '--', *DIRECTORIES], cwd=root, text=True).split('\0')
    return sorted(name for name in names if name.endswith(SUFFIXES) and not name.startswith(EXCLUDED))


def require_version(executable):
    try:
        version = subprocess.run([executable, '--version'], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'clang-format {CLANG_FORMAT_VERSION} is required: {error}')
    if not re.fullmatch(rf'clang-format version {re.escape(CLANG_FORMAT_VERSION)}{WHEEL_PROVENANCE}\n?', version):
        raise SystemExit(f'clang-format {CLANG_FORMAT_VERSION} is required; found: {version.strip()}')
    return executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clang-format', default='clang-format', help='clang-format executable')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--fix', action='store_true', help='rewrite the files in place')
    mode.add_argument('--list', action='store_true', help='print the checked files and exit')
    args = parser.parse_args()
    names = sources()
    if not names:
        raise SystemExit('no C or C++ sources discovered')
    if args.list:
        print('\n'.join(names))
        return 0
    # A path names a file relative to the invoking directory, not to ROOT, where clang-format runs.
    tool = require_version(args.clang_format if Path(args.clang_format).name == args.clang_format
                           else str(Path(args.clang_format).absolute()))
    flags = ['-i'] if args.fix else ['--dry-run', '--Werror']
    return subprocess.run([tool, '--style=file', *flags, *names], cwd=ROOT).returncode


if __name__ == '__main__':
    sys.exit(main())
