#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the pinned static LiveService transport closure for native macOS."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time

# Immutable upstream Git objects (release tags) of the versions deps/live-services-target.packman.xml pins.
PINS = {
    'abseil': ('https://github.com/abseil/abseil-cpp.git', '9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f'),
    'zlib': ('https://github.com/madler/zlib.git', '216c70c020aa53f0c40920d155f808b6b59c9acb'),
    'openssl': ('https://github.com/openssl/openssl.git', '45e844fa2a14ec92d146bd8f5778ac130b6625fb'),
    'c-ares': ('https://github.com/c-ares/c-ares.git', '0b88066f79ac9745d6819899f1794df91b73c873'),
    're2': ('https://github.com/google/re2.git', '4fe134adfc3d42cb728f2fbe1be0d89396361975'),
    'protobuf': ('https://github.com/protocolbuffers/protobuf.git', '2c32dede124c41fc815db054a8efe45c624ee1cb'),
    'grpc': ('https://github.com/grpc/grpc.git', 'aae267021b1ac256f8b9038d0ef528c3798cc137'),
}


# Installed outputs required by the downstream transport build.
REQUIRED_ARTIFACTS = {
    'abseil': ('include/absl/base/config.h', 'lib/libabsl_base.a'),
    'zlib': ('include/zlib.h', 'lib/libz.a'),
    'openssl': ('include/openssl/ssl.h', 'lib/libssl.a', 'lib/libcrypto.a', 'bin/openssl'),
    'c-ares': ('include/ares.h', 'lib/libcares.a'),
    're2': ('include/re2/re2.h', 'lib/libre2.a'),
    'protobuf': ('include/google/protobuf/message.h', 'lib/libprotobuf.a', 'lib/libprotoc.a', 'bin/protoc'),
    'grpc': ('include/grpcpp/grpcpp.h', 'lib/libgrpc.a', 'lib/libgrpc++.a', 'bin/grpc_cpp_plugin'),
}


def installed_artifacts(install, name):
    for relative in REQUIRED_ARTIFACTS[name]:
        path = install / relative
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Missing or empty installed transport artifact: {path}')
    paths = sorted(path for directory in ('include', 'lib', 'bin')
                   for path in (install / directory).rglob('*')
                   if path.is_file() and path.name != 'PACKAGE-INFO.yaml')
    result = {}
    for path in paths:
        relative = path.relative_to(install).as_posix()
        data = path.read_bytes()
        if (path.suffix == '.a' or relative.startswith('bin/')) and str(install.parent.resolve()).encode() in data:
            raise ValueError(f'Installed transport artifact names a build-machine path: {path}')
        result[relative] = hashlib.sha256(data).hexdigest()
    return result


def run(command, cwd=None):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def cmake_options(name, root):
    options = {
        'abseil': ['-DABSL_PROPAGATE_CXX_STD=ON', '-DABSL_BUILD_TESTING=OFF'],
        'zlib': ['-DZLIB_BUILD_TESTING=OFF', '-DZLIB_BUILD_SHARED=OFF', '-DZLIB_BUILD_STATIC=ON'],
        'c-ares': ['-DCARES_STATIC=ON', '-DCARES_SHARED=OFF', '-DCARES_BUILD_TESTS=OFF', '-DHAVE_PIPE2=OFF'],
        're2': ['-DRE2_BUILD_TESTING=OFF'],
        'protobuf': ['-Dprotobuf_BUILD_TESTS=OFF', '-Dprotobuf_BUILD_SHARED_LIBS=OFF',
                     '-Dprotobuf_ABSL_PROVIDER=package'],
        'grpc': ['-DgRPC_INSTALL=ON', '-DgRPC_BUILD_TESTS=OFF', '-DgRPC_DOWNLOAD_ARCHIVES=OFF', '-DgRPC_ABSL_PROVIDER=package',
                 '-DgRPC_CARES_PROVIDER=package', '-DgRPC_PROTOBUF_PROVIDER=package',
                 '-DgRPC_RE2_PROVIDER=package', '-DgRPC_SSL_PROVIDER=package',
                 '-DgRPC_ZLIB_PROVIDER=package', '-DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF',
                 '-DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF', '-DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF',
                 '-DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF', '-DgRPC_BUILD_GRPC_PYTHON_PLUGIN=OFF',
                 '-DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF',
                 f'-DOPENSSL_ROOT_DIR={root / "openssl-release"}'],
    }
    return options[name]


def provision(root, cmake, jobs, arch, *, reuse_only=False):
    if arch != 'arm64' or jobs < 1:
        raise ValueError('The transport build requires arm64 and a positive job count')
    root.mkdir(parents=True, exist_ok=True)
    recipe = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    toolchain = {tool: subprocess.check_output([tool, '--version'], text=True).splitlines()[0]
                 for tool in (str(cmake), 'clang')}
    toolchain.update(macos=platform.mac_ver()[0],
                     xcode=subprocess.check_output(['xcodebuild', '-version'], text=True).strip(),
                     sdk=subprocess.check_output(['xcrun', '--show-sdk-version'], text=True).strip())
    prefixes = ';'.join(str(root / f'{name}-release') for name in PINS)
    for name, (url, pin) in PINS.items():
        source = root / 'source' / name
        build = root / 'build' / name
        install = root / f'{name}-release'
        marker = install / 'source-pin.json'
        identity = {'url': url, 'git_object': pin, 'arch': arch, 'deployment_target': '13.0',
                    'configuration': 'Release', 'recipe_sha256': recipe, 'toolchain': toolchain}
        if reuse_only and not (source / '.git').is_dir():
            raise ValueError(f'Cached transport source checkout is missing: {name}')
        if (source / '.git').exists():
            dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
            if dirty.strip():
                raise ValueError(f'Dependency source has local changes: {source}')
        completed = json.loads(marker.read_text()) if marker.exists() else {}
        if completed.get('identity') == identity:
            head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
            expected = subprocess.check_output(['git', '-C', str(source), 'rev-parse', f'{pin}^{{commit}}'], text=True).strip()
            if head != expected:
                raise ValueError(f'Dependency source differs from its pin: {source}')
            if installed_artifacts(install, name) != completed.get('artifacts'):
                raise ValueError(f'Installed transport artifacts differ from their completion marker: {install}')
            continue
        if reuse_only:
            raise ValueError(f'No matching completed transport SDK is available for reuse: {name}')
        if not (source / '.git').exists():
            source.mkdir(parents=True, exist_ok=True)
            run(['git', 'init', source])
            run(['git', '-C', source, 'remote', 'add', 'origin', url])
        run(['git', '-C', source, 'fetch', '--depth=1', 'origin', pin])
        run(['git', '-C', source, 'checkout', '--detach', 'FETCH_HEAD'])
        if install.exists():
            retired = root / 'retired'
            retired.mkdir(exist_ok=True)
            install.rename(retired / f'{name}-{time.time_ns()}')
        if build.exists():
            retired = root / 'retired'
            retired.mkdir(exist_ok=True)
            build.rename(retired / f'{name}-build-{time.time_ns()}')
        build.mkdir(parents=True, exist_ok=True)
        path_map = f'-ffile-prefix-map={root.resolve()}=transport'
        if name == 'openssl':
            target = 'darwin64-arm64-cc'
            # Runtime defaults use a logical system prefix; DESTDIR stages the SDK privately.
            # OpenSSL compiles relative source paths and records its compiler flags as runtime text.
            openssl_path_map = '-ffile-prefix-map=../../source/openssl=transport/openssl'
            run([source / 'Configure', target, 'no-shared', 'no-tests',
                 '--prefix=/usr/local', '--openssldir=/etc/ssl', '--libdir=lib',
                 '-mmacosx-version-min=13.0', openssl_path_map], cwd=build)
            run(['make', f'-j{jobs}'], cwd=build)
            staged = build / 'package-root'
            run(['make', 'install_sw', f'DESTDIR={staged}'], cwd=build)
            (staged / 'usr/local').rename(install)
        else:
            run([cmake, '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                 '-DCMAKE_CXX_STANDARD=17', '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
                 '-DBUILD_SHARED_LIBS=OFF', '-DCMAKE_INSTALL_LIBDIR=lib',
                 f'-DCMAKE_C_FLAGS={path_map}', f'-DCMAKE_CXX_FLAGS={path_map}',
                 f'-DCMAKE_OSX_ARCHITECTURES={arch}', '-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0',
                 f'-DCMAKE_INSTALL_PREFIX={install}', f'-DCMAKE_PREFIX_PATH={prefixes}',
                 *cmake_options(name, root)])
            run([cmake, '--build', build, '--parallel', str(jobs)])
            run([cmake, '--install', build])
        completed = dict(identity=identity, artifacts=installed_artifacts(install, name))
        marker.write_text(json.dumps(completed, indent=2) + '\n')
    run([root / 'protobuf-release/bin/protoc', '--version'])
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.cache/mac-transport'))
    parser.add_argument('--cmake', default='cmake')
    parser.add_argument('--parallel', type=int, default=4)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('A native arm64 macOS host is required')
    if args.parallel < 1:
        parser.error('--parallel must be positive')
    provision(args.output.resolve(), args.cmake, args.parallel, platform.machine())


if __name__ == '__main__':
    main()
