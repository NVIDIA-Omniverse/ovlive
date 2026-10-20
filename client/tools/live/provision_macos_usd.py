#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a pinned stock USD 25.11 and TBB SDK for a native macOS arm64 host."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

SOURCES = {
    "OpenUSD": ("https://github.com/PixarAnimationStudios/OpenUSD.git", "363a7c8da8d1937072a5f0989e91faf72eb1fa76"),
    "oneTBB": ("https://github.com/oneapi-src/oneTBB.git", "9afd759b72c0c233cd5ea3c3c06b0894c9da9c54"),
}


def sdk_identity(python_root, cmake):
    return dict(sources={name: dict(url=url, commit=pin) for name, (url, pin) in SOURCES.items()},
                recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                python_root=str(python_root), architecture='arm64', configuration='Release',
                tools={tool: subprocess.check_output(command, text=True).strip() for tool, command in (
                    ('python', [str(python_root / 'bin/python3.12'), '--version']),
                    ('cmake', [str(cmake), '--version']), ('clang', ['clang', '--version']),
                    ('macos_sdk', ['xcrun', '--show-sdk-version']))})


def installed_artifacts(prefix):
    return {path.relative_to(prefix).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ('include', 'lib', 'bin') for path in sorted((prefix / directory).rglob('*'))
            if path.is_file() and path.name != 'PACKAGE-INFO.yaml'}


def run(command):
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def checkout(directory, url, revision, *, reuse_only=False):
    if not directory.exists():
        if reuse_only:
            raise ValueError(f'Cached source checkout is missing: {directory}')
        run(["git", "init", directory])
    else:
        if not (directory / '.git').is_dir():
            raise ValueError(f'Source checkout is not a Git repository: {directory}')
        try:
            actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        except subprocess.CalledProcessError:
            # Only a symbolic branch whose ref is absent identifies an unborn checkout.
            reference = subprocess.check_output(["git", "-C", str(directory), "symbolic-ref", "--quiet", "HEAD"],
                                                text=True).strip()
            missing = subprocess.run(["git", "-C", str(directory), "show-ref", "--verify", "--quiet", reference],
                                     check=False)
            if not reference.startswith('refs/heads/') or missing.returncode != 1:
                raise
            actual = None
        dirty = subprocess.check_output(["git", "--no-optional-locks", "-C", str(directory),
                                         "status", "--porcelain"], text=True).strip()
        if dirty or (reuse_only and actual != revision):
            raise ValueError(f"Source checkout must be clean at {revision}: {directory}")
        if actual == revision:
            return
    run(["git", "-C", directory, "fetch", "--depth=1", url, revision])
    run(["git", "-C", directory, "checkout", "--detach", "FETCH_HEAD"])
    actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "--no-optional-locks", "-C", str(directory),
                                     "status", "--porcelain"], text=True).strip()
    if actual != revision or dirty:
        raise ValueError(f"Source checkout must be clean at {revision}: {directory}")


def validate_sdk(prefix, python, python_root):
    for relative in ('include/pxr/pxr.h', 'include/oneapi/tbb.h', 'lib/libtbb.dylib',
                     'lib/python/pxr/Usd/__init__.py'):
        path = prefix / relative
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Missing or empty installed USD SDK artifact: {path}')
    libraries = sorted((prefix / 'lib').glob('*usd*.dylib'))
    if not libraries:
        raise ValueError('Installed USD SDK contains no USD libraries')
    for library in libraries:
        arch = subprocess.check_output(['lipo', '-archs', str(library)], text=True).strip()
        if arch != 'arm64':
            raise ValueError(f'Installed USD SDK library is not arm64: {library}')
    environment = dict(os.environ, PYTHONPATH=str(prefix / 'lib/python'),
                       DYLD_LIBRARY_PATH=os.pathsep.join((str(prefix / 'lib'), str(python_root / 'lib'))))
    subprocess.run([str(python), '-c',
                    'from pxr import Usd; assert Usd.GetVersion() == (0, 25, 11); '
                    'assert Usd.Stage.CreateInMemory()'], env=environment, check=True)


def publish_sdk(root, prefix):
    # The public selection changes only after the fresh SDK passes validation.
    selected = root / 'install'
    with tempfile.TemporaryDirectory(prefix='.install-next-', dir=root) as temporary:
        pending = Path(temporary) / 'install'
        # The relative target names the public selection's parent after replacement.
        pending.symlink_to(prefix.relative_to(root), target_is_directory=True)
        if selected.exists() and not selected.is_symlink():
            if not selected.is_dir():
                raise ValueError(f'SDK selection is not an install directory: {selected}')
            retired = root / 'retired'
            retired.mkdir(exist_ok=True)
            selected.rename(retired / f'install-{time.time_ns()}')
        pending.replace(selected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3] / ".cache/mac-sdk")
    parser.add_argument("--python-root", required=True, type=Path, help="Python 3.12 framework version directory")
    parser.add_argument("--cmake", required=True, type=Path)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--reuse-only", action="store_true", help="validate the selected cached SDK without building")
    args = parser.parse_args(argv)
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        parser.error("requires a native macOS arm64 host")
    if not 1 <= args.parallel <= 64:
        parser.error("--parallel must be between 1 and 64")
    root, python_root, cmake = args.root.resolve(), args.python_root.resolve(), args.cmake.absolute()
    python = python_root / "bin/python3.12"
    version = subprocess.check_output([str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"], text=True).strip()
    if version != "3.12" or not (python_root / "Python").is_file():
        parser.error("--python-root must name a Python 3.12 framework SDK")
    root.mkdir(parents=True, exist_ok=True)
    identity = sdk_identity(python_root, cmake)
    for name, (url, revision) in SOURCES.items():
        if args.reuse_only and not (root / name / '.git').is_dir():
            raise ValueError(f'Cached USD source checkout is missing: {name}')
        checkout(root / name, url, revision, reuse_only=args.reuse_only)
    selected = root / 'install'
    if selected.is_symlink() and selected.exists():
        prefix = selected.resolve()
        if not prefix.is_relative_to(root / 'builds'):
            raise ValueError('Selected SDK must belong to this SDK build directory')
        record = json.loads((prefix.parent / 'build-provenance.json').read_text())
        if record.get('identity') == identity and record.get('complete') is True:
            if record.get('installed_artifacts') != installed_artifacts(prefix):
                raise ValueError('Cached USD SDK artifacts differ from build provenance')
            validate_sdk(prefix, python, python_root)
            print(f'Reuse validated stock USD SDK: {selected}')
            return
    if args.reuse_only:
        raise ValueError('No matching completed stock USD SDK is available for reuse')
    builds = root / "builds"
    builds.mkdir(exist_ok=True)
    fresh = Path(tempfile.mkdtemp(prefix="sdk-", dir=builds))
    prefix = fresh / "install"
    common = ["-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={prefix}",
              "-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_INSTALL_LIBDIR=lib"]
    commands = []
    for name, settings in (
        ("oneTBB", ["-DTBB_TEST=OFF", "-DTBB_STRICT=OFF"]),
        ("OpenUSD", [f"-DCMAKE_PREFIX_PATH={prefix}", f"-DTBB_ROOT_DIR={prefix}",
                     "-DPXR_BUILD_MONOLITHIC=OFF", "-DPXR_ENABLE_PYTHON_SUPPORT=ON",
                     "-DPXR_BUILD_IMAGING=OFF", "-DPXR_BUILD_USD_IMAGING=OFF", "-DPXR_BUILD_USDVIEW=OFF",
                     "-DPXR_BUILD_TESTS=OFF", "-DPXR_BUILD_EXAMPLES=OFF", "-DPXR_BUILD_TUTORIALS=OFF",
                     "-DPXR_BUILD_DOCUMENTATION=OFF", "-DPXR_BUILD_USD_TOOLS=OFF",
                     f"-DPython3_EXECUTABLE={python}", f"-DPython3_INCLUDE_DIR={python_root / 'include/python3.12'}",
                     f"-DPython3_LIBRARY={python_root / 'lib/libpython3.12.dylib'}"]),
    ):
        build = fresh / (name + "-build")
        commands.extend([[cmake, "-S", root / name, "-B", build, *common, *settings],
                         [cmake, "--build", build, "--parallel", str(args.parallel)],
                         [cmake, "--install", build]])
    record = {"identity": identity, "complete": False,
              "sources": {name: {"url": url, "commit": revision} for name, (url, revision) in SOURCES.items()},
              "python_root": str(python_root), "commands": [[str(value) for value in command] for command in commands]}
    (fresh / "build-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    for command in commands:
        run(command)
    validate_sdk(prefix, python, python_root)
    record.update(complete=True, installed_artifacts=installed_artifacts(prefix))
    (fresh / 'build-provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    publish_sdk(root, prefix)
    print(f"Stock USD SDK: {root / 'install'}")


if __name__ == "__main__":
    main()
