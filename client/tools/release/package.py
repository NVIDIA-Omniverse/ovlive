#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-pack a checked release archive as the packman package a Kit dependency entry consumes.

The package root is the verified `omni_usd_live` tree of the archive plus its `release-manifest.json`, extended
with the package name and version, so a packman `linkPath` named `omni_usd_live` is the Kit extension folder.
The package is a zip, which packman resolves like a 7z and the standard library verifies; beside it is a
`sha256sum` checksum file and nothing else, because both USD profiles' packages share one candidate directory.
GitHub renames release assets with special characters, so the file is `<name>--<version>.zip` and
`packman_filename` gives the `<name>@<version>.zip` that packman stores and resolves. The package is the one binary
release asset of its target, so `check` verifies it alone against its checksum file and the manifest it carries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live import build, ci, platforms  # noqa: E402
from tools.release import archive  # noqa: E402

# The vendored packman launcher: a batch file on Windows, a shell script elsewhere.
PACKMAN = ROOT / ('tools/packman/packman.cmd' if sys.platform == 'win32' else 'tools/packman/packman')
MANIFEST = 'release-manifest.json'
NAMES = {'kit-25.11': 'omni_usd_live_ov_openusd_0.25.11_py_3.12',
         'stock-25.11': 'omni_usd_live_openusd_0.25.11_py_3.12'}


def identity(record):
    """The packman name and version of one release manifest."""
    if record['usd_profile'] not in NAMES:
        raise ValueError('No package name for USD profile ' + record['usd_profile'])
    if not platforms.supported(record['platform']):
        raise ValueError('No package ABI for platform ' + str(record['platform']))
    # The extension version, tagged or not: a candidate qualifies main before its tag exists and finalize promotes
    # the same bytes.
    abi = platforms.get(record['platform']).dependency_abi
    return NAMES[record['usd_profile']], f"{record['version']}.{abi}.{record['configuration']}"


def asset_filename(name, version):
    """The GitHub-safe release asset name of a packman package."""
    return f'{name}--{version}.zip'


def packman_filename(asset):
    """The packman file name `<name>@<version>.zip` of a release asset name."""
    if not asset.endswith('.zip') or asset.count('--') != 1:
        raise ValueError('Not a packman package asset name: ' + asset)
    return asset.replace('--', '@')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value):
    return json.dumps(value, indent=2, sort_keys=True) + '\n'


def create(archive_path, output_dir, *, check=archive.check, run=subprocess.run):
    """Check `archive_path`, pack its package tree with packman into `output_dir`; return the `.zip` path."""
    archive_path, output_dir = Path(archive_path), Path(output_dir)
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        record = check(archive_path, work / 'extract', archive.machine_paths())
        if record['dirty']:
            raise ValueError('The archive was built from a modified checkout; release only a clean checkout')
        name, version = identity(record)
        filename = asset_filename(name, version)
        outputs = [output_dir / filename, output_dir / (filename + '.sha256')]
        existing = [path.name for path in outputs if path.exists() or path.is_symlink()]
        if existing:
            raise ValueError('Release package output exists: ' + ', '.join(existing))
        release = dict(record, package=dict(name=name, version=version))
        tree = work / name / version
        shutil.copytree(work / 'extract' / record['archive_root'] / archive.EXTENSION_FOLDER, tree)
        (tree / MANIFEST).write_text(_json(release))
        output_dir.mkdir(parents=True, exist_ok=True)
        run([str(PACKMAN), 'pack', '-c', 'zip', '-o', str(output_dir), str(tree)], check=True)
    packed = output_dir / packman_filename(filename)
    if not packed.is_file():
        raise ValueError('packman pack did not write ' + packed.name)
    path = packed.rename(output_dir / filename)
    (output_dir / (filename + '.sha256')).write_text(f'{_sha256(path)}  {filename}\n')
    return path


def _safe_name(name):
    parts = name.removesuffix('/').split('/')
    if name.startswith('/') or '\\' in name or any(part in ('..', '.', '') for part in parts):
        raise ValueError('Unsafe package member name: ' + name)


def check(path):
    """Verify a package against its required `.sha256` checksum file and its release manifest; return the manifest.

    The manifest is path-free and names this package's identity and file name; the files are exactly the manifest's
    package files beside the manifest, each with the SHA-256 and size the manifest records. The package alone binds
    its members to the release manifest; the candidate manifest and evidence bind the package bytes to the
    qualified native archive.
    """
    path = Path(path)
    checksum = path.with_name(path.name + '.sha256')
    if not checksum.is_file() or checksum.read_text(encoding='utf-8') != f'{_sha256(path)}  {path.name}\n':
        raise ValueError('Package checksum mismatch or missing checksum file: ' + path.name)
    with zipfile.ZipFile(path) as reader:
        infos = reader.infolist()
        for info in infos:
            _safe_name(info.filename)
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate package member in ' + path.name)
        files = {info.filename for info in infos if not info.is_dir()}
        if MANIFEST not in files:
            raise ValueError('Package has no ' + MANIFEST + ': ' + path.name)
        if reader.getinfo(MANIFEST).file_size > archive.MAX_MANIFEST_BYTES:
            raise ValueError('Release manifest is too large')
        record = json.loads(reader.read(MANIFEST), object_pairs_hook=ci.unique_object)
        if not isinstance(record, dict) or not isinstance(record.get('files'), dict):
            raise ValueError('Unrecognized release manifest in ' + path.name)
        archive._path_free(record)
        platform = record.get('platform')
        if (record.get('schema_version') != 1 or record.get('product') != 'ovlive'
                or record.get('extension_folder') != archive.EXTENSION_FOLDER
                or record.get('configuration') != 'release' or not platforms.supported(platform)
                or not re.fullmatch(r'\d+\.\d+\.\d+', str(record.get('version')))
                or record.get('usd_profile') not in build.USD_PROFILES
                or record.get('archive_root') != archive.archive_root(record['version'], record['usd_profile'], platform)):
            raise ValueError('Unrecognized release manifest in ' + path.name)
        name, version = identity(record)
        if record.get('package') != dict(name=name, version=version) or path.name != asset_filename(name, version):
            raise ValueError(f'The release manifest names another package name or version than {path.name}')
        package_files = build.package_files(platform)
        prefix = archive.EXTENSION_FOLDER + '/'
        if set(record['files']) != {prefix + member for member in package_files}:
            raise ValueError('The release manifest does not list exactly the package files')
        if files != {MANIFEST, *package_files}:
            raise ValueError(f'Unexpected or missing package members in {path.name}: '
                             f'extra={sorted(files - {MANIFEST, *package_files})}, '
                             f'missing={sorted(set(package_files) - files)}')
        parents = {str(parent) for member in files for parent in PurePosixPath(member).parents if str(parent) != '.'}
        if not {info.filename.removesuffix('/') for info in infos if info.is_dir()} <= parents:
            raise ValueError('Unexpected package directories in ' + path.name)
        for member in sorted(package_files):
            data, entry = reader.read(member), record['files'][prefix + member]
            if (not isinstance(entry, dict) or entry.get('sha256') != hashlib.sha256(data).hexdigest()
                    or entry.get('size') != len(data)):
                raise ValueError('Package member differs from its manifest: ' + member)
        if record.get('components') != archive._components(reader.read('share/omni_usd_live/BOM.json')):
            raise ValueError('The release manifest components differ from the packaged bill of materials')
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='release archive tools/release/archive.py made')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        path = create(args.archive, args.output_dir)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'release package: {error}', file=sys.stderr)
        return 1
    print(f'{path.name}: {_sha256(path)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
