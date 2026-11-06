#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Package release-matched OSS sources and notices for a container's binary.

Run in the Debian Rust builder after compilation. `--binary NAME` names the
release binary the container ships as `/app/NAME` (default `live-edit`, the
service; `live-edit-conformance` for the conformance image): its SHA-256 goes
into SOURCE-MANIFEST.json and SHA256SUMS. Only explicit third-party inputs and
Cargo.lock are archived; the application checkout is never archived. Upstream
lockfiles in those trees are dropped before archiving, so the only lockfiles
among the archive's own files are the binary's Cargo.lock and the Rust
standard library's. Each file whose name looks like a lockfile is dropped when
it has a known lockfile name, kept when listed as not a lockfile, and
otherwise fails the packaging, naming the lists to edit. Archives inside it (FlatBuffers'
`git archive`, the standard library's `.crate` files, the GCC sources) are not
opened and keep any lockfiles of their own.
Debian .dsc fields follow https://www.debian.org/doc/debian-policy/ch-controlfields.html.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from email.parser import Parser
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tomllib
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen


CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"


def command(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, destination, checksum=None, size=None):
    destination = Path(destination)
    with urlopen(url, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    if size is not None and destination.stat().st_size != size:
        raise ValueError(f"Incorrect source size: {destination.name}")
    if checksum is not None and sha256(destination) != checksum:
        raise ValueError(f"Incorrect source checksum: {destination.name}")


def parse_dsc(text, package, version):
    # .dsc files may be OpenPGP clear-signed; preserve the signed original in
    # the bundle, but parse only its Debian control stanza here.
    if text.startswith("-----BEGIN PGP SIGNED MESSAGE-----"):
        text = text.split("\n\n", 1)[1].split("-----BEGIN PGP SIGNATURE-----", 1)[0]
        text = "\n".join(
            line[2:] if line.startswith("- ") else line for line in text.splitlines()
        )
    fields = Parser().parsestr(text)
    if fields.get("Source") != package or fields.get("Version") != version:
        raise ValueError("Debian source package does not match installed libstdc++")
    files = []
    for line in fields.get("Checksums-Sha256", "").splitlines():
        if not line.strip():
            continue
        checksum, size, name = line.split()
        if not re.fullmatch(r"[0-9a-f]{64}", checksum) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9.+_~-]*", name
        ):
            raise ValueError("Invalid Debian source checksum or filename")
        files.append((name, checksum, int(size)))
    if not files:
        raise ValueError("Debian source descriptor has no SHA-256 file inventory")
    return files


def collect_gcc(destination, licenses):
    library = command("g++", "-print-file-name=libstdc++.a")
    if not Path(library).is_file():
        raise ValueError("The linked static libstdc++ archive is unavailable")
    owner = command("dpkg-query", "-S", library).rsplit(": ", 1)[0]
    package, version = command(
        "dpkg-query", "-W", "-f=${source:Package}\t${source:Version}", owner
    ).split("\t")
    if not re.fullmatch(r"gcc-\d+", package):
        raise ValueError(f"Unsupported libstdc++ source package: {package}")
    name = f"{package}_{version.split(':')[-1]}.dsc"
    destination.mkdir()
    # The installed Debian package identifies the exact source revision,
    # including Debian patches. An upstream GCC version alone is insufficient.
    for base in (
        "https://deb.debian.org/debian/pool/main",
        "https://security.debian.org/debian-security/pool/updates/main",
    ):
        directory = f"{base}/g/{package}"
        try:
            download(f"{directory}/{quote(name)}", destination / name)
            break
        except HTTPError as error:
            if error.code != 404:
                raise
    else:
        raise ValueError(f"Source unavailable for installed {package}={version}")
    for filename, checksum, size in parse_dsc(
        (destination / name).read_text(), package, version
    ):
        download(
            f"{directory}/{quote(filename)}", destination / filename, checksum, size
        )
    copyright_file = Path("/usr/share/doc") / owner.split(":")[0] / "copyright"
    shutil.copyfile(copyright_file, licenses / "GCC-COPYRIGHT.txt")
    shutil.copyfile("/usr/share/common-licenses/GPL-3", licenses / "GPL-3.txt")
    return {
        "package": package,
        "version": version,
        "source_url": f"{directory}/{quote(name)}",
    }


def collect_rust(sysroot, destination, licenses):
    """Copy the compiler's rust-src tree and fetch the crates its lock pins.

    The copy is whole; `drop_upstream_lockfiles` later removes the lockfiles
    of the source tree's sub-projects, keeping only `library/Cargo.lock`; the
    fetched `.crate` archives are kept as published.
    """
    source = sysroot / "lib/rustlib/src/rust"
    lock = source / "library/Cargo.lock"
    if not lock.is_file():
        raise ValueError("Install rust-src for the compiler used to build this binary")
    shutil.copytree(source, destination / "source")
    shutil.copyfile(
        sysroot / "share/doc/rust/COPYRIGHT-library.html",
        licenses / "RUST-COPYRIGHT.html",
    )
    shutil.copytree(sysroot / "share/doc/rust/licenses", licenses / "rust-licenses")
    archives = destination / "dependencies"
    archives.mkdir()
    packages = tomllib.loads(lock.read_text())["package"]

    def fetch(package):
        if "source" not in package:
            return
        if package["source"] != CRATES_IO:
            raise ValueError(
                f"Unsupported Rust runtime dependency source: {package['source']}"
            )
        name, version = package["name"], package["version"]
        filename = f"{name}-{version}.crate"
        download(
            f"https://static.crates.io/crates/{name}/{filename}",
            archives / filename,
            package["checksum"],
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(fetch, packages))


#: The records each patched crate tree ships beside it, as `<tree><suffix>`.
PATCHED_CRATE_RECORDS = (".patch", ".SOURCE.md", ".SHA256SUMS")


def collect_patched_crates(project, manifest, destination):
    """Copy each crate the root manifest patches to a vendored path.

    `cargo vendor` copies registry and git packages only, so a crate that
    `[patch.crates-io]` points at a local tree — the copy compiled into the
    binary — would otherwise be missing from the sources. Each tree ships with
    the patch applied to the published crate (`<tree>.patch`), the record of
    its provenance (`<tree>.SOURCE.md`) and the digests of the published
    crate's files that record cites (`<tree>.SHA256SUMS`); all three must
    exist. The tree is copied whole; `drop_upstream_lockfiles` later removes
    its lockfiles, which `<tree>.SHA256SUMS` still lists because it records
    the published release.
    """
    patches = manifest.get("patch", {}).get("crates-io", {})
    trees = sorted(spec["path"] for spec in patches.values() if "path" in spec)
    if not trees:
        return
    destination.mkdir()
    for tree in trees:
        source = project / tree
        if not source.is_dir():
            raise ValueError(f"Patched crate tree is missing: {tree}")
        shutil.copytree(source, destination / source.name)
        for suffix in PATCHED_CRATE_RECORDS:
            record = source.with_name(source.name + suffix)
            if not record.is_file():
                raise ValueError(f"Patched crate record is missing: {record.name}")
            shutil.copyfile(record, destination / record.name)


#: The lockfile detector: what might be a lockfile, searched for in a file's
#: name, case-insensitively. An upstream tree's lockfile pins the versions its
#: own developers resolved, not the ones this binary was built from, so a
#: scanner reading it would report versions the image does not contain. The
#: detector is deliberately broader than KNOWN_LOCKFILE_NAMES, so that a
#: lockfile variant no list of names foresaw stops the packaging (see
#: `drop_upstream_lockfiles`) and fails `check-container-oss.sh` instead of
#: shipping unnoticed. It matches a name that
#: - ends in `.lock`, `.lockb` or `.lockfile` (`Cargo.lock`, `yarn.lock`,
#:   `MODULE.bazel.lock`, `bun.lockb`, `gradle.lockfile`);
#: - has `.lock.` in it (`Cargo.lock.old`, `packages.lock.json`);
#: - is `lock`, or ends in `-lock`, `_lock` or `.lock`, before `.json`,
#:   `.yaml`, `.yml` or `.toml` (`package-lock.json`, `pnpm-lock.yaml`), the
#:   separator keeping words like `clock.json` or `deadlock.yaml` out;
#: - is `pylock.toml` or `pylock.<name>.toml` (Python's standard lockfile);
#: - ends in `shrinkwrap.json`, `shrinkwrap.yaml` or `shrinkwrap.yml`
#:   (`npm-shrinkwrap.json`);
#: - is `Package.resolved` (Swift Package Manager).
#: Manifests (`Cargo.toml`, `package.json`), `go.sum` and source files named
#: for locks (`rwlock.rs`, `lock.rs`, `spin-lock.h`) do not match. It is
#: written in lower case and in POSIX ERE syntax, with `^` only at the start of
#: an alternative: `check-container-oss.sh` applies the same text to the
#: lower-cased name of each archive member.
LOCKFILE_PATTERN = re.compile(
    r"\.lock(b|file)?$|\.lock\.|^lock\.(json|ya?ml|toml)$"
    r"|[-_.]lock\.(json|ya?ml|toml)$|^pylock\.(.*\.)?toml$"
    r"|shrinkwrap\.(json|ya?ml)$|^package\.resolved$",
    re.IGNORECASE,
)

#: The file names decided to be dependency lockfiles, as exact names or
#: `fnmatch` globs over a file name, matched case-sensitively. Each must also
#: match LOCKFILE_PATTERN. A detector match named here is dropped from the
#: archive unless it is one of the KEPT_LOCKFILES. To classify a newly
#: detected lockfile, add its name, or a glob for its family, here.
KNOWN_LOCKFILE_NAMES = frozenset(
    {
        "Cargo.lock",
        "Cargo.lock.*",  # Cargo.lock.old, Cargo.lock.orig: a renamed Cargo lock
        "Cargo-*.lock",  # a Cargo lock kept under another name
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "shrinkwrap.yaml",  # pnpm's former lockfile name
        "bun.lock",
        "bun.lockb",
        "deno.lock",
        "Gemfile.lock",
        "Podfile.lock",
        "Pipfile.lock",
        "poetry.lock",
        "uv.lock",
        "pdm.lock",
        "pylock.toml",
        "pylock.*.toml",
        "pixi.lock",
        "conda-lock.yml",
        "composer.lock",
        "packages.lock.json",
        "gradle.lockfile",
        "pubspec.lock",
        "Package.resolved",
        "mix.lock",
        "conan.lock",
        "flake.lock",
        "MODULE.bazel.lock",
    }
)

#: Detector matches that are not lockfiles and ship, as paths relative to
#: `sources/` (an exact path, so a crate upgrade is classified afresh). Empty:
#: no file in the archive's trees is one. To add an entry, put the path here
#: with a comment saying what the file is, and the same path under `sources/`
#: in `not_lockfiles` in `check-container-oss.sh`, which otherwise rejects it.
NOT_LOCKFILES = frozenset()

#: The lockfiles the archive keeps, relative to `sources/`: the binary's
#: resolution, and the standard library's, which pins the crates in
#: `sources/rust/dependencies`. `check-container-oss.sh` requires exactly these
#: lockfile members, each once, under `sources/`.
KEPT_LOCKFILES = frozenset({"Cargo.lock", "rust/source/library/Cargo.lock"})


def is_known_lockfile(name):
    """Whether a file named `name` is a lockfile KNOWN_LOCKFILE_NAMES lists."""
    return any(fnmatch.fnmatchcase(name, known) for known in KNOWN_LOCKFILE_NAMES)


def lockfile_origin(relative):
    """The tree a file at `relative` (a path under `sources/`) came from."""
    if relative.parts[0] in ("crates", "patched-crates") and len(relative.parts) > 2:
        kind = "vendored" if relative.parts[0] == "crates" else "patched"
        return f"{kind} crate {relative.parts[1]}"
    if len(relative.parts) == 1:
        return "the top of sources/"
    return f"the {relative.parts[0]} tree"


def drop_upstream_lockfiles(sources):
    """Delete the upstream lockfiles from the `sources` work tree, so the
    archive carries no member `check-container-oss.sh` would reject.

    Each file whose name LOCKFILE_PATTERN matches is classified: one of the
    KEPT_LOCKFILES or NOT_LOCKFILES stays; one whose name KNOWN_LOCKFILE_NAMES
    lists is deleted. Any other match fails the packaging before anything is
    deleted, naming every such file, so a new kind of file is classified by
    a person rather than deleted from the sources or shipped unchecked.

    `cargo vendor` lists every file of a vendored crate, its lockfile
    included, with its SHA-256 in the crate's `.cargo-checksum.json`, and
    cargo verifies each listed file when it builds from that directory. So a
    lockfile removed from `crates/<crate>/` is also removed from that crate's
    `"files"` map, and the map is written back in cargo's own serialization;
    `"package"`, the published `.crate`'s SHA-256, is unchanged. A lockfile
    the map does not list fails the packaging rather than shipping a crate
    directory cargo would reject. Patched crate trees are path sources, which
    cargo does not verify, so they carry no such map. It matches files by name
    and never opens an archive in the tree (`flatbuffers.tar`, the `.crate`
    files, the GCC sources), so lockfiles inside those stay.
    """
    dropped, unclassified = [], []
    for path in sorted(sources.rglob("*")):
        relative = path.relative_to(sources)
        if (
            not LOCKFILE_PATTERN.search(path.name)
            or not path.is_file()
            or relative.as_posix() in KEPT_LOCKFILES | NOT_LOCKFILES
        ):
            continue
        if is_known_lockfile(path.name):
            dropped.append((path, relative))
        else:
            unclassified.append(f"  {relative.as_posix()} ({lockfile_origin(relative)})")
    if unclassified:
        raise ValueError(
            "Unclassified lockfile-like files under sources/ (LOCKFILE_PATTERN "
            "matches their names):\n"
            + "\n".join(unclassified)
            + "\nIf a file is a dependency lockfile, add its name to "
            "KNOWN_LOCKFILE_NAMES in package_oss.py and it is dropped. If it is "
            "not, add its path under sources/ to NOT_LOCKFILES in package_oss.py "
            "and, prefixed with sources/, to not_lockfiles in "
            "check-container-oss.sh, and it ships."
        )
    checksums = {}
    for path, relative in dropped:
        if relative.parts[0] == "crates":
            crate = relative.parts[1]
            key = Path(*relative.parts[2:]).as_posix()
            if crate not in checksums:
                checksums[crate] = json.loads(
                    (sources / "crates" / crate / ".cargo-checksum.json").read_text(
                        encoding="utf-8"
                    )
                )
            files = checksums[crate]["files"]
            if key not in files:
                raise ValueError(
                    f"Vendored crate {crate} does not list {key} in .cargo-checksum.json"
                )
            del files[key]
        path.unlink()
    for crate, data in checksums.items():
        (sources / "crates" / crate / ".cargo-checksum.json").write_text(
            json.dumps(data, separators=(",", ":"), sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )


def archive_sources(source, destination):
    def readable(member):
        # Source checkouts can be created with restrictive umasks. Users must
        # be able to read/extract the archive without root.
        member.uid = member.gid = 0
        member.uname = member.gname = "root"
        if member.issym() or member.islnk():
            raise ValueError(f"Unexpected source symlink: {member.name}")
        member.mode = 0o755 if member.isdir() or member.mode & 0o111 else 0o644
        return member

    with tarfile.open(destination, "w:gz") as archive:
        archive.add(source, arcname="sources", filter=readable)


#: The service's binary: the default `--binary`, and the only one that links
#: jemalloc (`src/allocator.rs` is in the service binary alone).
SERVICE_BINARY = "live-edit"


def package_oss(project, flatbuffers, output, work, binary=SERVICE_BINARY):
    # Refuse reuse so a previous build can never contribute stale source files.
    output.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    sources = work / "sources"
    sources.mkdir()
    licenses = output / "licenses"
    licenses.mkdir()
    manifest = tomllib.loads((project / "Cargo.toml").read_text())
    release = manifest["package"]["version"]
    shutil.copyfile(project / "ATTRIBUTIONS.md", licenses / "ATTRIBUTIONS.md")
    shutil.copyfile(project / "LICENSE.txt", licenses / "LICENSE.txt")
    shutil.copyfile(project / "NOTICE", licenses / "NOTICE")
    shutil.copyfile(project / "Cargo.lock", sources / "Cargo.lock")
    subprocess.run(
        ["cargo", "vendor", "--locked", "--versioned-dirs", str(sources / "crates")],
        cwd=project,
        stdout=subprocess.DEVNULL,
        check=True,
    )
    if (project / "Cargo.lock").read_bytes() != (sources / "Cargo.lock").read_bytes():
        raise ValueError("Cargo.lock changed while collecting release sources")
    collect_patched_crates(project, manifest, sources / "patched-crates")

    flatbuffers_version = command(
        "git", "describe", "--tags", "--exact-match", "HEAD", cwd=flatbuffers
    )
    flatbuffers_commit = command("git", "rev-parse", "HEAD", cwd=flatbuffers)
    subprocess.run(
        [
            "git",
            "-C",
            str(flatbuffers),
            "archive",
            "--format=tar",
            "--prefix=flatbuffers/",
            f"--output={sources / 'flatbuffers.tar'}",
            "HEAD",
        ],
        check=True,
    )
    shutil.copyfile(flatbuffers / "LICENSE", licenses / "FLATBUFFERS-LICENSE.txt")
    header_notice = (
        flatbuffers / "include/flatbuffers/flatbuffers.h"
    ).read_text().split("*/", 1)[0] + "*/\n"
    (licenses / "FLATBUFFERS-NOTICE.txt").write_text(header_notice)

    rust_version = command("rustc", "--version", "--verbose")
    collect_rust(
        Path(command("rustc", "--print", "sysroot")), sources / "rust", licenses
    )
    gcc = collect_gcc(sources / "gcc", licenses)
    # Every tree is in place: drop the upstream lockfiles before anything below
    # describes or archives the work tree.
    drop_upstream_lockfiles(sources)
    metadata = {
        "service_version": release,
        "binary_sha256": sha256(project / "target/release" / binary),
        "cargo_lock_sha256": sha256(sources / "Cargo.lock"),
        "rustc": rust_version,
        "flatbuffers": {"version": flatbuffers_version, "commit": flatbuffers_commit},
        "gcc": gcc,
    }
    (sources / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copyfile(sources / "manifest.json", licenses / "SOURCE-MANIFEST.json")
    # Only the service binary links jemalloc; the archive vendors the whole
    # lock either way, so only this paragraph depends on the binary.
    allocator = (
        "The service's memory allocator is jemalloc, whose C sources are compiled "
        "and statically linked into the binary by the `tikv-jemalloc-sys` crate. Its "
        "source is `sources/crates/tikv-jemalloc-sys-<version>/jemalloc`, and its "
        "BSD-2-Clause notice is `jemalloc/COPYING` in that crate directory, reproduced "
        "in ATTRIBUTIONS.md under `tikv-jemalloc-sys`.\n\n"
        if binary == SERVICE_BINARY
        else ""
    )
    (licenses / "SOURCES.md").write_text(
        f"# Live Edit {release}: third-party sources and notices\n\n"
        "`/vendor.tar.gz` contains the locked Cargo dependency sources (`sources/crates`), "
        "FlatBuffers (`sources/flatbuffers.tar`), Rust standard-library sources and their "
        "locked dependency archives (`sources/rust`), and the exact Debian GCC source "
        "package for the statically linked C++ runtime (`sources/gcc`). "
        "Use `dpkg-source -x <file>.dsc` to unpack the GCC source and Debian patches.\n\n"
        f"{allocator}"
        "A crate built from a locally patched copy of its published release is in "
        "`sources/patched-crates/<name>-<version>` instead of `sources/crates`, without "
        "its lockfiles. Beside it are the patch (`<name>-<version>.patch`), the record of "
        "where it came from and why (`<name>-<version>.SOURCE.md`) and the SHA-256 of "
        "each file of the published release (`<name>-<version>.SHA256SUMS`). Its notice is in "
        "ATTRIBUTIONS.md.\n\n"
        "Upstream lockfiles are left out of the archive's own files: they pin the "
        "versions each project's own developers resolved, not the ones the binary was "
        "built from. A lockfile here is a dependency lockfile, such as `Cargo.lock` "
        "or a renamed copy of one (`Cargo.lock.old`), `yarn.lock`, `package-lock.json`, "
        "`npm-shrinkwrap.json`, `flake.lock` or `MODULE.bazel.lock`. Lockfiles are "
        "recognised by file name, so a test input named like one is removed too; a "
        "file whose name only resembles a lockfile ships. "
        "The only lockfiles among the archive's own files are `sources/Cargo.lock`, "
        "the binary's, and `sources/rust/source/library/Cargo.lock`, the standard "
        "library's. Archives inside the archive are kept as they are: "
        "`sources/flatbuffers.tar` is `git archive` of the FlatBuffers commit "
        "`manifest.json` records, and the `.crate` files in `sources/rust/dependencies` "
        "and the files in `sources/gcc` are the published downloads, so any lockfiles "
        "inside them remain. A vendored crate's `.cargo-checksum.json` does not list "
        "the lockfiles removed from it, so `sources/crates` works as a Cargo vendored "
        "source.\n\n"
        "The bundle is a source superset: Cargo build/test/other-platform dependencies "
        "are retained, but NVIDIA application source is not included.\n\n"
        "Cargo notices: ATTRIBUTIONS.md. FlatBuffers: FLATBUFFERS-NOTICE.txt and "
        "FLATBUFFERS-LICENSE.txt. Rust runtime: RUST-COPYRIGHT.html and rust-licenses/. "
        "GCC runtime: GCC-COPYRIGHT.txt and GPL-3.txt (including the runtime exception "
        "in GCC-COPYRIGHT.txt). SOURCE-MANIFEST.json identifies the build inputs.\n\n"
        "The NVIDIA distroless base supplies its own notices. Its source index is "
        "https://developer.nvidia.com/w/distroless-oss/cc/index.html .\n"
    )
    shutil.copytree(licenses, sources / "licenses")
    archive_sources(sources, output / "vendor.tar.gz")
    files = sorted(p for p in output.rglob("*") if p.is_file())
    checksums = [f"{sha256(path)}  {path.relative_to(output)}\n" for path in files]
    checksums.append(f"{metadata['binary_sha256']}  app/{binary}\n")
    (licenses / "SHA256SUMS").write_text("".join(checksums))
    for path in output.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--flatbuffers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--binary",
        default=SERVICE_BINARY,
        help="the release binary the container ships as /app/NAME, read from "
        "target/release/NAME; its SHA-256 goes into SOURCE-MANIFEST.json and "
        "SHA256SUMS (default: %(default)s, the service; the jemalloc paragraph "
        "of SOURCES.md is written only for it)",
        metavar="NAME",
    )
    args = parser.parse_args()
    package_oss(
        args.project.resolve(),
        args.flatbuffers.resolve(),
        args.output.resolve(),
        args.work.resolve(),
        args.binary,
    )
