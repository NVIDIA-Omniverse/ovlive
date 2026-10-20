#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create or check the reproducible release archive of a validated `release` stage.

`create` archives the exact package of a `tools/live/build.py` stage under `<root>/omni_usd_live/`, beside a path-free
`<root>/release-manifest.json`, and writes a `sha256sum` checksum file; `check` verifies an archive member by member
and, with `--against`, compares it member by member with a second archive of the same name.
The archive carries no package-manifest.json and no dependency metadata.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live import build, platforms  # noqa: E402
from tools.live.test import fingerprint  # noqa: E402

EXTENSION_FOLDER = 'omni_usd_live'
MANIFEST = 'release-manifest.json'
SUFFIX = '.tar.gz'
FILE_MODES = (0o644, 0o755)
MAX_MANIFEST_BYTES = 1024 * 1024
# The prebuilt static dependencies record their own build machine's paths under this prefix; one such path ends at
# the first NUL, whitespace, quote, `=`, `:` or `;`.
PREBUILT_PREFIX = '/builds/omniverse'
PREBUILT_BUILD_PATHS = re.compile(re.escape(PREBUILT_PREFIX).encode() + rb'[^\0\s"\'=:;]*')
# A forbidden path matches only as a whole path or a path prefix: no path character precedes it, and no path
# character other than `/` follows it.
PATH_BEFORE = rb'(?<![A-Za-z0-9._+/~-])'
PATH_AFTER = rb'(?![A-Za-z0-9._+~-])'
# Standard system prefixes hold compilers and build tools on every machine; they name no machine.
SYSTEM_ROOTS = {'/', '/usr', '/usr/local', '/bin', '/opt'}


def version(path=None):
    """The `[package]` table's `version` of the Kit extension configuration, `MAJOR.MINOR.PATCH`."""
    text = Path(path or ROOT / 'source/kit/config/extension.toml').read_text(encoding='utf-8')
    table = re.search(r'^\[package\][ \t]*\n(.*?)(?=^\[|\Z)', text, re.M | re.S)
    found = re.findall(r'^version[ \t]*=[ \t]*"([^"\n]*)"[ \t]*$', table.group(1) if table else '', re.M)
    if len(found) != 1 or not re.fullmatch(r'\d+\.\d+\.\d+', found[0]):
        raise ValueError('The extension configuration needs one [package] version = "MAJOR.MINOR.PATCH"')
    return found[0]


def _git(*arguments):
    result = subprocess.run(['git', *arguments], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise ValueError(f'git {arguments[0]} failed: ' + (result.stderr.strip().splitlines() or [''])[0])
    return result.stdout


def git_head():
    return _git('rev-parse', 'HEAD').strip()


def git_dirty():
    return bool(_git('status', '--porcelain').strip())


def git_changes():
    """The count of modified, staged and untracked non-ignored files in the checkout."""
    return len(_git('status', '--porcelain', '--untracked-files=all').splitlines())


def git_tags():
    return _git('tag', '--points-at', 'HEAD').split()


def commit_time():
    """Archive member time: `SOURCE_DATE_EPOCH` when set, else the head commit's time."""
    value = os.environ.get('SOURCE_DATE_EPOCH') or _git('show', '-s', '--format=%ct', 'HEAD').strip()
    if not re.fullmatch(r'\d+', value):
        raise ValueError('The archive time must be a nonnegative integer')
    return int(value)


def archive_root(release_version, profile, platform):
    return f'ovlive-{release_version}-{profile}-{platforms.get(platform).name}'


def _mode(name, platform):
    return 0o755 if name in build.native_files(platform) else 0o644


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def _path_free(record):
    absolute = sorted(value for value in _strings(record) if value.startswith('/'))
    if absolute:
        raise ValueError('The release manifest must not hold an absolute path: ' + absolute[0])


def _components(bom_bytes):
    bom = json.loads(bom_bytes)
    if not isinstance(bom, dict) or bom.get('schema_version') != 1 or not isinstance(bom.get('components'), list):
        raise ValueError('Unrecognized packaged bill of materials')
    return bom['components']


def manifest(package, record, head, tag, dirty):
    """The path-free release manifest of one validated stage."""
    package = Path(package)
    identity = record['support_identity']
    profile, platform = identity['usd_profile'], record['platform']
    release_version = version()
    library = (package / build.library(platform)).read_bytes()
    if not re.fullmatch(r'[0-9a-f]{40}', head) or head.encode() not in library:
        raise ValueError('The library does not embed the source revision ' + head + '; rebuild the stage')
    abi = re.search(r'^#define OMNI_USD_LIVE_ABI_VERSION UINT32_C\((0x[0-9A-Fa-f]{8})\)$',
                    (package / 'include/omni/usd/live/Live.h').read_text(encoding='utf-8'), re.M)
    if not abi:
        raise ValueError('The packaged C header does not define its ABI version')
    files = {}
    for name in build.package_files(platform):
        path = package / name
        files[f'{EXTENSION_FOLDER}/{name}'] = dict(sha256=build._hash(path), size=path.stat().st_size,
                                                   mode=f'{_mode(name, platform):04o}')
    result = dict(
        schema_version=1, product='ovlive', version=release_version,
        archive_root=archive_root(release_version, profile, platform), extension_folder=EXTENSION_FOLDER,
        platform=platform, configuration=record['configuration'], usd_profile=profile, sanitizer='none',
        abi_version=abi.group(1).lower(), git_head=head, git_tag=tag, dirty=dirty,
        source_fingerprint=fingerprint(),
        source_inputs_sha256=hashlib.sha256(
            json.dumps(record['inputs'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        embedded_revision=head, stripped=False,
        sdk_identity={key: value['sha256'] for key, value in identity['trees'].items()},
        tool_identity={key: value['sha256'] for key, value in identity['tools'].items()},
        components=_components((package / 'share/omni_usd_live/BOM.json').read_bytes()),
        files=files)
    _path_free(result)
    return result


def _recorded(value):
    """A path as a stage record holds it: a POSIX path stays POSIX off a POSIX host, which cannot resolve it."""
    if os.name != 'posix' and isinstance(value, str) and value.startswith('/'):
        return PurePosixPath(value)
    return Path(value)


def machine_paths(record=None):
    """The checkout, SDK roots and tool roots of a stage record, as selected and resolved; always this checkout.

    The checkout is the repository root, the client root's parent, so the repository's `.cache` is covered too.
    """
    support = (record or {}).get('support_inputs') or {}
    roots = [ROOT.parent, support.get('root'), *(support.get('sdks') or {}).values()]
    for tool in (support.get('tools') or {}).values():
        parent = _recorded(tool).parent
        roots.append(parent.parent if parent.name == 'bin' else parent)
    paths = set()
    for value in roots:
        if value:
            path = value if isinstance(value, PurePosixPath) else _recorded(value)
            for spelling in ((path, path.resolve()) if isinstance(path, Path) else (path,)):
                if str(spelling) not in SYSTEM_ROOTS:
                    paths.add(str(spelling))
    return sorted(paths)


def _info(name, kind, mode, mtime, size=0):
    info = tarfile.TarInfo(name)
    info.type, info.mode, info.mtime, info.size = kind, mode, mtime, size
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    return info


def create(package, output_dir, *, allow_dirty=False, extract_to=None):
    """Archive a validated `release` stage of this host's platform reproducibly, then check the archive; return its path."""
    package, output_dir = Path(package), Path(output_dir)
    record = build.validate_stage(package, ROOT, expected_platform=platforms.host().name,
                                  expected_configuration='release')
    release_version = version()
    head, tags, dirty = git_head(), git_tags(), git_dirty()
    releases = sorted(tag for tag in tags if re.fullmatch(r'v\d+\.\d+\.\d+', tag))
    if any(tag != 'v' + release_version for tag in releases):
        raise ValueError(f'Head tag {", ".join(releases)} does not match the extension version {release_version}')
    if dirty and not allow_dirty:
        raise ValueError('The checkout is modified; commit it or pass --allow-dirty for a development archive')
    record_manifest = manifest(package, record, head, releases[0] if releases else None, dirty)
    root = record_manifest['archive_root']
    name = root + SUFFIX
    target, checksum, temporary = (output_dir / name, output_dir / (name + '.sha256'),
                                   output_dir / (name + '.tmp'))
    if any(path.exists() or path.is_symlink() for path in (target, checksum, temporary)):
        raise ValueError('Release archive output exists: ' + name)
    if extract_to is not None and (Path(extract_to).exists() or Path(extract_to).is_symlink()):
        raise ValueError('Extraction directory exists: ' + str(extract_to))
    output_dir.mkdir(parents=True, exist_ok=True)
    mtime = commit_time()
    manifest_bytes = (json.dumps(record_manifest, indent=2, sort_keys=True) + '\n').encode()
    members = {root: None, f'{root}/{MANIFEST}': manifest_bytes}
    for member in build.package_files(record['platform']):
        members[f'{root}/{EXTENSION_FOLDER}/{member}'] = package / member
        for parent in PurePosixPath(f'{root}/{EXTENSION_FOLDER}/{member}').parents:
            if str(parent) != '.':
                members.setdefault(str(parent), None)
    digest = hashlib.sha256()
    published = made = False
    try:
        with temporary.open('xb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed, \
                    tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as writer:
                for member in sorted(members, key=lambda item: PurePosixPath(item).parts):
                    source = members[member]
                    if source is None:
                        writer.addfile(_info(member, tarfile.DIRTYPE, 0o755, mtime))
                        continue
                    data = source if isinstance(source, bytes) else source.read_bytes()
                    mode = 0o644 if isinstance(source, bytes) else _mode(member.split('/', 2)[2], record['platform'])
                    writer.addfile(_info(member, tarfile.REGTYPE, mode, mtime, len(data)), io.BytesIO(data))
        with temporary.open('rb') as written:
            for chunk in iter(lambda: written.read(1024 * 1024), b''):
                digest.update(chunk)
        # The archive is checked under its temporary name; only a passing archive is published.
        if extract_to is not None:
            Path(extract_to).mkdir(parents=True, exist_ok=False)
            made = True
        _verify(temporary, root, extract_to, machine_paths(record), made=made)
        temporary.rename(target)
        published = True
        checksum.write_text(f'{digest.hexdigest()}  {name}\n')
    except BaseException:
        temporary.unlink(missing_ok=True)
        if published:
            target.unlink(missing_ok=True)
            checksum.unlink(missing_ok=True)
        if made:
            shutil.rmtree(extract_to)
        raise
    return target


def _reject(name, reason):
    raise ValueError(f'Unsafe or unexpected release archive member {name}: {reason}')


def check(archive, extract_to=None, forbidden=None):
    """Verify an archive against its required `<archive>.sha256` checksum file and its manifest; return the manifest.

    No member may contain a `forbidden` path (by default this checkout's) outside the prebuilt dependencies' own
    `/builds/omniverse` build paths.
    """
    archive = Path(archive)
    if not archive.name.endswith(SUFFIX):
        raise ValueError('A release archive is named <root>' + SUFFIX)
    root = archive.name[:-len(SUFFIX)]
    checksum = archive.with_name(archive.name + '.sha256')
    if not checksum.is_file():
        raise ValueError('Release archive checksum file is missing: ' + checksum.name)
    digest = build._hash(archive)
    if checksum.read_text(encoding='utf-8') != f'{digest}  {archive.name}\n':
        raise ValueError('Release archive checksum mismatch: ' + archive.name)
    return _verify(archive, root, extract_to, machine_paths() if forbidden is None else forbidden)


def _identities(archive):
    """Each member's header fields and content SHA-256, by name."""
    with tarfile.open(archive, 'r:gz') as reader:
        return {info.name: (info.type, info.mode, info.mtime, info.uid, info.gid, info.uname, info.gname, info.size,
                            hashlib.sha256(reader.extractfile(info).read()).hexdigest() if info.isfile() else None)
                for info in reader.getmembers()}


def compare(archive, other, extract_to=None, forbidden=None):
    """Check two archives of one root, then compare them member by member.

    Returns the sorted names of the members whose header or content differ, the sorted top-level release manifest
    fields other than `files` that differ, and whether the archive files are byte-identical.
    """
    archive, other = Path(archive), Path(other)
    if archive.name != other.name:
        raise ValueError(f'Release archives of different roots: {archive.name} and {other.name}')
    record = check(archive, extract_to, forbidden)
    other_record = check(other, forbidden=forbidden)
    first, second = _identities(archive), _identities(other)
    members = sorted((name for name in first.keys() | second.keys() if first.get(name) != second.get(name)),
                     key=lambda item: PurePosixPath(item).parts)
    fields = sorted(key for key in record.keys() | other_record.keys()
                    if key != 'files' and record.get(key) != other_record.get(key))
    return members, fields, build._hash(archive) == build._hash(other)


def _verify(archive, root, extract_to, forbidden, *, made=False):
    """The member checks of `check`; `made` says the caller created the empty `extract_to`."""
    paths = [str(path).rstrip('/') for path in forbidden if str(path).rstrip('/')]
    shadowed = sorted(path for path in paths if path.startswith(PREBUILT_PREFIX))
    if shadowed:
        raise ValueError(f'A forbidden path lies under the prebuilt dependency build paths: {shadowed[0]}')
    leak = re.compile(PATH_BEFORE + b'(' + b'|'.join(re.escape(path.encode()) for path in sorted(paths, key=len,
                      reverse=True)) + b')' + PATH_AFTER) if paths else None
    with tarfile.open(archive, 'r:gz') as reader:
        members = {}
        for info in reader.getmembers():
            name = info.name
            parts = PurePosixPath(name).parts
            if not (info.isfile() or info.isdir()):
                _reject(name, 'not a regular file or directory')
            if name.startswith('/'):
                _reject(name, 'absolute name')
            if '\\' in name:
                _reject(name, 'backslash in name')
            if any(part in ('..', '.', '') for part in name.split('/')):
                _reject(name, 'parent-relative or unnormalized name')
            if parts[0] != root:
                _reject(name, 'outside ' + root)
            if name in members:
                _reject(name, 'duplicate')
            if info.mode not in (FILE_MODES if info.isfile() else (0o755,)):
                _reject(name, f'mode {info.mode:04o}')
            members[name] = info
        manifest_name = f'{root}/{MANIFEST}'
        if manifest_name not in members or not members[manifest_name].isfile():
            raise ValueError('Release archive has no ' + MANIFEST)
        if members[manifest_name].size > MAX_MANIFEST_BYTES:
            raise ValueError('Release manifest is too large')
        record = json.loads(reader.extractfile(members[manifest_name]).read())
        if not isinstance(record, dict) or not isinstance(record.get('files'), dict):
            raise ValueError('Unrecognized release manifest')
        _path_free(record)
        if (record.get('schema_version') != 1 or record.get('product') != 'ovlive'
                or record.get('extension_folder') != EXTENSION_FOLDER or record.get('configuration') != 'release'
                or not platforms.supported(record.get('platform')) or record.get('archive_root') != root
                or not re.fullmatch(r'\d+\.\d+\.\d+', str(record.get('version')))
                or record.get('usd_profile') not in build.USD_PROFILES
                or archive_root(record['version'], record['usd_profile'], record['platform']) != root):
            raise ValueError('The release manifest does not match the archive name ' + archive.name)
        files = record['files']
        platform = record['platform']
        expected = {f'{EXTENSION_FOLDER}/{name}' for name in build.package_files(platform)}
        if set(files) != expected:
            raise ValueError('The release manifest does not list exactly the package files')
        actual = {name[len(root) + 1:] for name, info in members.items() if info.isfile()}
        if actual != {MANIFEST} | expected:
            raise ValueError(f'Unexpected or missing release archive members: '
                             f'extra={sorted(actual - expected - {MANIFEST})}, missing={sorted(expected - actual)}')
        directories = {str(parent) for name in actual for parent in PurePosixPath(f'{root}/{name}').parents
                       if str(parent) != '.'}
        if {name for name, info in members.items() if info.isdir()} != directories:
            raise ValueError('Unexpected release archive directories')
        for name in sorted(files):
            info, entry = members[f'{root}/{name}'], files[name]
            data = reader.extractfile(info).read()
            if (entry != dict(sha256=hashlib.sha256(data).hexdigest(), size=len(data), mode=f'{info.mode:04o}')
                    or info.size != len(data)):
                raise ValueError('Release archive member differs from its manifest: ' + name)
            found = leak.search(PREBUILT_BUILD_PATHS.sub(b'', data)) if leak else None
            if found:
                raise ValueError(f'Release archive member {name} names a path of the build machine: '
                                 + found.group(1).decode(errors='replace'))
            if info.mode != _mode(name.split('/', 1)[1], platform):
                _reject(name, f'mode {info.mode:04o}')
        bom = reader.extractfile(members[f'{root}/{EXTENSION_FOLDER}/share/omni_usd_live/BOM.json']).read()
        if record.get('components') != _components(bom):
            raise ValueError('The release manifest components differ from the packaged bill of materials')
        if extract_to is not None:
            extract_to = Path(extract_to)
            if not made:
                if extract_to.exists() or extract_to.is_symlink():
                    raise ValueError('Extraction directory exists: ' + str(extract_to))
                extract_to.mkdir(parents=True)
            reader.extractall(extract_to, filter='data')
    if extract_to is not None:
        build._validate_layout(extract_to / root / EXTENSION_FOLDER, platform, exact=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create_parser = commands.add_parser('create', help='archive a validated release stage')
    create_parser.add_argument('--package', type=Path, required=True, help='<target-dir>/usd/omniverse/live')
    create_parser.add_argument('--output-dir', type=Path, required=True)
    create_parser.add_argument('--allow-dirty', action='store_true',
                               help='archive a modified checkout, recorded as dirty')
    create_parser.add_argument('--extract-to', type=Path, help='fresh directory to extract and validate into')
    check_parser = commands.add_parser('check', help='verify a release archive member by member')
    check_parser.add_argument('--archive', type=Path, required=True)
    check_parser.add_argument('--extract-to', type=Path, help='fresh directory to extract and validate into')
    check_parser.add_argument('--forbid', action='append', default=[], metavar='PATH',
                              help='a build machine path no member may contain, besides this checkout')
    check_parser.add_argument('--against', type=Path, metavar='ARCHIVE',
                              help='a second archive of the same name, checked too, whose members must be identical')
    args = parser.parse_args(argv)
    try:
        if args.command == 'create':
            path = create(args.package, args.output_dir, allow_dirty=args.allow_dirty, extract_to=args.extract_to)
            print(f'{path.name}: {build._hash(path)}')
        elif args.against is not None:
            members, fields, identical = compare(args.archive, args.against, args.extract_to,
                                                 forbidden=machine_paths() + args.forbid)
            for name in members:
                print('differing member: ' + name)
            for field in fields:
                print('differing release manifest field: ' + field)
            print(f'{args.archive.name}: {len(members)} differing members, archive files '
                  + ('byte-identical' if identical else 'differ'))
            return 0 if identical and not members else 1
        else:
            check(args.archive, args.extract_to, forbidden=machine_paths() + args.forbid)
            print(f'{args.archive.name}: OK')
    except (ValueError, OSError, EOFError, tarfile.TarError) as error:
        print(f'release archive: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
