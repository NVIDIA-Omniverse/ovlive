#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prune obsolete completed USD SDK installs after provisioning validates the selected SDK."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def generated_name(name):
    return re.fullmatch(r'sdk-[A-Za-z0-9_-]+', name) is not None


def completed_install(build, *, exact=False):
    """Recognize only completed generated installs whose recorded files remain exact."""
    prefix, marker = build / 'install', build / 'build-provenance.json'
    if build.is_symlink() or not build.is_dir() or prefix.is_symlink() or not prefix.is_dir():
        raise ValueError('SDK build and install must be real directories')
    if marker.is_symlink() or not marker.is_file():
        raise ValueError('SDK provenance must be a regular file')
    record = json.loads(marker.read_text())
    if not isinstance(record, dict) or record.get('complete') is not True:
        raise ValueError('SDK provenance must record a completed build')
    identity = record.get('identity')
    if (not isinstance(identity, dict) or identity.get('architecture') != 'arm64'
            or identity.get('configuration') != 'Release'
            or not isinstance(identity.get('recipe_sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', identity['recipe_sha256'])):
        raise ValueError('SDK provenance must identify a generated arm64 release build')
    artifacts = record.get('installed_artifacts')
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError('SDK provenance must record installed artifacts')
    directories = {'include', 'lib', 'bin'}
    for name, digest in artifacts.items():
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or path.as_posix() != name
                or len(path.parts) < 2 or path.parts[0] not in directories
                or path.name == 'PACKAGE-INFO.yaml' or not isinstance(digest, str)
                or not re.fullmatch(r'[0-9a-f]{64}', digest)):
            raise ValueError('SDK artifact inventory is invalid')
        directories.update(parent.as_posix() for parent in path.parents if parent != Path('.'))
    for relative, digest in artifacts.items():
        path = prefix / relative
        if not path.is_file() or not path.resolve(strict=True).is_relative_to(prefix):
            raise ValueError('SDK artifact must belong to its install')
        if path.resolve(strict=True).name == 'PACKAGE-INFO.yaml':
            raise ValueError('SDK artifact cannot name dependency metadata')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('SDK install differs from its completed provenance')
    if not exact:
        return
    found = set()
    pending = []
    for directory in ('include', 'lib', 'bin'):
        path = prefix / directory
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError('SDK artifact directories must be real directories')
        if path.is_dir():
            pending.append(path)
    while pending:
        for path in pending.pop().iterdir():
            relative = path.relative_to(prefix).as_posix()
            if path.name == 'PACKAGE-INFO.yaml':
                raise ValueError('SDK install contains untracked metadata')
            if path.is_symlink():
                target = path.resolve(strict=True)
                if (not target.is_relative_to(prefix) or not target.is_file()
                        or target.name == 'PACKAGE-INFO.yaml'):
                    raise ValueError('SDK install contains an unexpected symlink')
            elif path.is_dir():
                if relative not in directories:
                    raise ValueError('SDK install contains an untracked directory')
                pending.append(path)
                continue
            elif not path.is_file():
                raise ValueError('SDK install contains an unexpected entry')
            if relative not in artifacts:
                raise ValueError('SDK install contains an untracked file')
            found.add(relative)
    if found != set(artifacts):
        raise ValueError('SDK install differs from its completed provenance')


def prune(root):
    """Keep the selected install and preserve every entry that lacks exact generated provenance."""
    root = Path(root).absolute()
    builds, selected = root / 'builds', root / 'install'
    if root.is_symlink() or not root.is_dir() or builds.is_symlink() or not builds.is_dir():
        raise ValueError('SDK root and builds must be real directories')
    root = root.resolve(strict=True)
    builds, selected = root / 'builds', root / 'install'
    if not selected.is_symlink():
        raise ValueError('Selected SDK must be a generated install symlink')
    target = selected.readlink()
    if (target.is_absolute() or len(target.parts) != 3 or target.parts[0] != 'builds'
            or not generated_name(target.parts[1]) or target.parts[2] != 'install'):
        raise ValueError('Selected SDK must name builds/sdk-*/install directly')
    active = builds / target.parts[1]
    completed_install(active)
    obsolete = []
    for build in sorted(builds.iterdir()):
        if build == active or not generated_name(build.name) or build.is_symlink() or not build.is_dir():
            continue
        try:
            completed_install(build, exact=True)
        except (ValueError, FileNotFoundError, RuntimeError):
            continue
        obsolete.append(build)
    if selected.readlink() != target:
        raise ValueError('Selected SDK changed during cache pruning')
    for build in obsolete:
        prefix = build / 'install'
        for name in ('include', 'lib', 'bin'):
            if (prefix / name).is_dir():
                shutil.rmtree(prefix / name)
        (build / 'build-provenance.json').unlink()
        if not any(prefix.iterdir()):
            prefix.rmdir()
        if not any(build.iterdir()):
            build.rmdir()
    return [build.name for build in obsolete]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3] / '.cache/mac-sdk')
    args = parser.parse_args(argv)
    try:
        removed = prune(args.root)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f'Pruned {len(removed)} obsolete completed USD SDK installs')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
