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

"""Package release-matched OSS sources and notices for the service container.

Run in the Debian Rust builder after compilation. Only explicit third-party
inputs and Cargo.lock are archived; the application checkout is never archived.
Debian .dsc fields follow https://www.debian.org/doc/debian-policy/ch-controlfields.html.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from email.parser import Parser
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
    exist.
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


def package_oss(project, flatbuffers, output, work):
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
    metadata = {
        "service_version": release,
        "binary_sha256": sha256(project / "target/release/live-edit"),
        "cargo_lock_sha256": sha256(sources / "Cargo.lock"),
        "rustc": rust_version,
        "flatbuffers": {"version": flatbuffers_version, "commit": flatbuffers_commit},
        "gcc": gcc,
    }
    (sources / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copyfile(sources / "manifest.json", licenses / "SOURCE-MANIFEST.json")
    (licenses / "SOURCES.md").write_text(
        f"# Live Edit {release}: third-party sources and notices\n\n"
        "`/vendor.tar.gz` contains the locked Cargo dependency sources (`sources/crates`), "
        "FlatBuffers (`sources/flatbuffers.tar`), Rust standard-library sources and their "
        "locked dependency archives (`sources/rust`), and the exact Debian GCC source "
        "package for the statically linked C++ runtime (`sources/gcc`). "
        "Use `dpkg-source -x <file>.dsc` to unpack the GCC source and Debian patches.\n\n"
        "The service's memory allocator is jemalloc, whose C sources are compiled "
        "and statically linked into the binary by the `tikv-jemalloc-sys` crate. Its "
        "source is `sources/crates/tikv-jemalloc-sys-<version>/jemalloc`, and its "
        "BSD-2-Clause notice is `jemalloc/COPYING` in that crate directory, reproduced "
        "in ATTRIBUTIONS.md under `tikv-jemalloc-sys`.\n\n"
        "A crate built from a locally patched copy of its published release is in "
        "`sources/patched-crates/<name>-<version>` instead of `sources/crates`, beside "
        "the patch (`<name>-<version>.patch`), the record of where it came from "
        "and why (`<name>-<version>.SOURCE.md`) and the SHA-256 of each file of the "
        "published release (`<name>-<version>.SHA256SUMS`). Its notice is in "
        "ATTRIBUTIONS.md.\n\n"
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
    checksums.append(f"{metadata['binary_sha256']}  app/live-edit\n")
    (licenses / "SHA256SUMS").write_text("".join(checksums))
    for path in output.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--flatbuffers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    package_oss(
        args.project.resolve(),
        args.flatbuffers.resolve(),
        args.output.resolve(),
        args.work.resolve(),
    )
