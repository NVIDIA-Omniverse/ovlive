#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate or check THIRD_PARTY_LICENSES.txt and BOM.json from the pinned dependency license files.

Each statically linked package contributes exactly <root>/<package>-<config>/PACKAGE-LICENSES/<named file>; package
metadata beside it is never read. Versions come from deps/live-services-target.packman.xml and must match the resolved
package directory name. The outputs carry no local path and no package server URL.
A null BOM version means the component is host-provided and not pinned by this repository, or vendored source
without a release version. The vendored SHA-256 source (linkage `source`) has no recorded public upstream, so its
upstream is null and a provenance field names its origin; its license_sha256 is the SHA-256 of the concatenated MIT
and BSD-2-Clause texts in the THIRD_PARTY_NOTICES.md fenced blocks, which carry those texts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
PINS = 'deps/live-services-target.packman.xml'
LICENSES = 'THIRD_PARTY_LICENSES.txt'
BOM = 'BOM.json'
# package: (license file in PACKAGE-LICENSES, SPDX expression, public upstream project)
COMPONENTS = {
    'grpc': ('grpc-LICENSE.txt', 'Apache-2.0 AND BSD-3-Clause AND MPL-2.0', 'https://github.com/grpc/grpc'),
    'protobuf': ('protobuf-LICENSE.txt', 'BSD-3-Clause', 'https://github.com/protocolbuffers/protobuf'),
    'abseil': ('abseil-LICENSE.txt', 'Apache-2.0', 'https://github.com/abseil/abseil-cpp'),
    're2': ('re2-LICENSE.txt', 'BSD-3-Clause', 'https://github.com/google/re2'),
    'c-ares': ('c-ares-LICENSE.md', 'MIT', 'https://github.com/c-ares/c-ares'),
    # openssl-LICENSE (no extension) in the same directory is a build-time Perl module's license, not OpenSSL's.
    'openssl': ('openssl-LICENSE.txt', 'Apache-2.0', 'https://github.com/openssl/openssl'),
    'zlib': ('zlib-LICENSE.txt', 'Zlib', 'https://github.com/madler/zlib'),
}
FLATBUFFERS = ('flatbuffers', '1.12.0', 'Apache-2.0', 'https://github.com/google/flatbuffers',
               'source/live/wire/third_party/flatbuffers/LICENSE.txt')
PYBIND11 = ('pybind11', '2.11.1', 'BSD-3-Clause', 'https://github.com/pybind/pybind11',
            'PACKAGE-LICENSES/pybind11-LICENSE.txt')
# Host SDK shared libraries the package loads but does not redistribute.
HOST_SHARED = (('OpenUSD', '25.11', 'LicenseRef-OpenUSD', 'https://github.com/PixarAnimationStudios/OpenUSD'),
               ('TBB', None, 'Apache-2.0', 'https://github.com/uxlfoundation/oneTBB'),
               ('Python', '3.12', 'PSF-2.0', 'https://github.com/python/cpython'))
SHA256_SOURCE = dict(
    name='sha256', version=None, license='MIT AND BSD-2-Clause', upstream=None, linkage='source',
    provenance="source/live/support/SHA256.cpp, derived from Zilong Tan's SHA-256 implementation, "
               "itself derived from Allan Saddi's code; MIT and BSD-2-Clause texts in THIRD_PARTY_NOTICES.md")
PREAMBLE = (b'License texts of the third-party components compiled into the omni_usd_live package.\n'
            b'BOM.json lists each component with its version, license, upstream project and linkage.\n')


class InputError(Exception):
    """An input is missing or does not match the pinned dependency closure."""


def pins(root=ROOT):
    """Pinned packman version of each component, without the platform/configuration suffix."""
    versions = {}
    for package in ET.fromstring((root / PINS).read_text(encoding='utf-8')).iter('package'):
        version, separator, _ = package.get('version', '').partition('-${live_services_target_platform_abi}')
        if not separator or not version:
            raise InputError(f'Unrecognized packman version for {package.get("name")}')
        versions[package.get('name')] = version
    return versions


def _versioned(directory, version, name):
    real = directory.resolve(strict=True)
    if not real.is_dir() or not (real.name == version or real.name.startswith(version + '-')):
        raise InputError(f'{name}: resolved package directory {real.name} is not version {version}')


def _section(name, version, spdx, text):
    return (f'==== {name} {version} ({spdx}) ====\n\n'.encode('utf-8') + text
            + (b'' if text.endswith(b'\n') else b'\n'))


def _recorded_pybind11(output):
    """The pybind11 license text of the checked-in outputs, verified against the checked-in BOM hash."""
    name, version, spdx = PYBIND11[:3]
    try:
        bundle = (output / LICENSES).read_bytes()
        bom = json.loads((output / BOM).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise InputError(f'--pybind11-root is required without a recorded pybind11 license: {error}') from error
    recorded = [c.get('license_sha256') for c in bom.get('components', []) if c.get('name') == name]
    header = f'==== {name} {version} ({spdx}) ====\n\n'.encode('utf-8')
    if len(recorded) != 1 or bundle.count(header) != 1:
        raise InputError('--pybind11-root is required without a recorded pybind11 license')
    text = bundle.split(header, 1)[1]
    for candidate in (text, text[:-1]):
        if hashlib.sha256(candidate).hexdigest() == recorded[0]:
            return candidate
    raise InputError('The recorded pybind11 license differs from its recorded hash')


def _sha256_license(root):
    """The concatenated MIT and BSD-2-Clause texts of the SHA-256 section; `root` is the client root."""
    notices = (root / 'THIRD_PARTY_NOTICES.md').read_bytes()
    _, heading, section = notices.partition(b'## SHA-256 implementation (MIT and BSD-2-Clause)\n')
    section = section.split(b'\n## ', 1)[0]
    blocks = section.split(b'```text\n')[1:]
    if not heading or len(blocks) != 2 or any(b'```' not in block for block in blocks):
        raise InputError('THIRD_PARTY_NOTICES.md lacks the SHA-256 MIT and BSD-2-Clause texts')
    mit, bsd = [block.split(b'```', 1)[0] for block in blocks]
    if not mit.startswith(b'The MIT License\n') or bsd != (root / 'LICENSES/SHA256-BSD-2-Clause.txt').read_bytes():
        raise InputError('SHA-256 license text differs from its preserved notice')
    return mit + bsd


def generate(dependency_root, config, pybind11_root=None, output=ROOT, root=ROOT):
    """(THIRD_PARTY_LICENSES.txt bytes, BOM.json bytes); `root` is the client root."""
    dependency_root = Path(dependency_root)
    versions = pins(root)
    if set(versions) != set(COMPONENTS):
        raise InputError('Packman pins and the license component table differ')
    sections, components = [], []
    for name, (license_name, spdx, upstream) in COMPONENTS.items():
        package = dependency_root / f'{name}-{config}'
        _versioned(package, versions[name], name)
        text = (package / 'PACKAGE-LICENSES' / license_name).read_bytes()
        sections.append(_section(name, versions[name], spdx, text))
        components.append(dict(name=name, version=versions[name], license=spdx, upstream=upstream,
                               linkage='static', license_sha256=hashlib.sha256(text).hexdigest()))
    if pybind11_root is None:
        pybind11 = _recorded_pybind11(Path(output))
    else:
        _versioned(Path(pybind11_root), PYBIND11[1], PYBIND11[0])
        pybind11 = (Path(pybind11_root) / PYBIND11[4]).read_bytes()
    # pybind11 stays the last section: without its root, its recorded text runs to the end of the bundle.
    for (name, version, spdx, upstream, _), text in ((FLATBUFFERS, (root / FLATBUFFERS[4]).read_bytes()),
                                                     (PYBIND11, pybind11)):
        sections.append(_section(name, version, spdx, text))
        components.append(dict(name=name, version=version, license=spdx, upstream=upstream,
                               linkage='header', license_sha256=hashlib.sha256(text).hexdigest()))
    components.append(dict(SHA256_SOURCE, license_sha256=hashlib.sha256(_sha256_license(root)).hexdigest()))
    for name, version, spdx, upstream in HOST_SHARED:
        components.append(dict(name=name, version=version, license=spdx, upstream=upstream,
                               linkage='host-shared', license_sha256=None))
    bundle = PREAMBLE + b''.join(b'\n' + section for section in sections)
    bom = json.dumps(dict(schema_version=1, scope='ovlive client native package and host dependencies; excludes the Live Edit service',
                          components=components), indent=2) + '\n'
    return bundle, bom.encode('utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-root', type=Path, required=True,
                        help='directory holding the pinned <package>-<config> packman links')
    parser.add_argument('--config', choices=('debug', 'release'), required=True)
    parser.add_argument('--pybind11-root', type=Path,
                        help='pybind11 package root; without it the recorded pybind11 license is reused')
    parser.add_argument('--output-dir', type=Path, default=ROOT, help='directory of the checked-in outputs')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='regenerate in memory and compare with the outputs')
    mode.add_argument('--write', action='store_true', help='write the outputs')
    args = parser.parse_args(argv)
    try:
        bundle, bom = generate(args.dependency_root, args.config, args.pybind11_root, args.output_dir)
    except (InputError, OSError, ET.ParseError) as error:
        print(f'third_party_notices: {error}', file=sys.stderr)
        return 2
    outputs = {LICENSES: bundle, BOM: bom}
    if args.write:
        for name, content in outputs.items():
            (args.output_dir / name).write_bytes(content)
        return 0
    stale = []
    for name, content in outputs.items():
        try:
            if (args.output_dir / name).read_bytes() != content:
                stale.append(name)
        except OSError:
            stale.append(name)
    for name in stale:
        print(f'third_party_notices: {name} differs from the pinned license files; regenerate with --write',
              file=sys.stderr)
    if not stale:
        print('Third-party license bundle and bill of materials: OK')
    return 1 if stale else 0


if __name__ == '__main__':
    raise SystemExit(main())
