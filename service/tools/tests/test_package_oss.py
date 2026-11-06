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

"""Source identity, archive safety and final-image packaging regressions."""

from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools import package_oss as oss


ROOT = Path(__file__).resolve().parents[2]
#: The image gate, `check-container-oss.sh`.
GATE = ROOT / "tools/check-container-oss.sh"
# The vendored directory `cargo vendor --versioned-dirs` writes for the locked
# jemalloc binding. The real name, `+` build metadata included, so the gate's
# member pattern is exercised against what the release build actually vendors.
JEMALLOC_CRATE = (
    "tikv-jemalloc-sys-0.6.1+5.3.0-1-ge13ca993e8ccb9ba9847cc330696e02839f328f7"
)

#: The vendored crate the cargo stub writes, as `cargo vendor` would: its
#: files, upstream lockfiles among them, and a `.cargo-checksum.json` listing
#: each with its SHA-256. `book/package.json` is a manifest beside a lockfile,
#: `src/rwlock.rs` a source file named for a lock, the non-ASCII name holds
#: the map's serialization to cargo's, which does not escape it, and
#: `Cargo.lock.old` and `MODULE.bazel.lock` are lockfile variants real
#: vendored crates carry.
DEPENDENCY_CRATE = "dependency-1.0.0"
DEPENDENCY_FILES = {
    "Cargo.toml": b"third-party manifest",
    "lib.rs": b"third-party source",
    "src/rwlock.rs": b"third-party lock source",
    "Cargo.lock": b"upstream crate lock",
    "Cargo.lock.old": b"upstream renamed crate lock",
    "MODULE.bazel.lock": b"upstream Bazel lock",
    "book/package.json": b"upstream book manifest",
    "book/package-lock.json": b"upstream book lock",
    "docs/naïve.md": b"upstream docs",
}
#: The lockfiles among DEPENDENCY_FILES, which packaging drops.
DEPENDENCY_LOCKFILES = (
    "Cargo.lock",
    "Cargo.lock.old",
    "MODULE.bazel.lock",
    "book/package-lock.json",
)

#: A representative file name for each glob in KNOWN_LOCKFILE_NAMES; an exact
#: name stands for itself.
KNOWN_GLOB_INSTANCES = {
    "Cargo.lock.*": "Cargo.lock.old",
    "Cargo-*.lock": "Cargo-minimal.lock",
    "pylock.*.toml": "pylock.dev.toml",
}


def known_lockfile_instances(test):
    """One file name per KNOWN_LOCKFILE_NAMES entry, failing `test` for a glob
    KNOWN_GLOB_INSTANCES has no instance of."""
    instances = []
    for known in sorted(oss.KNOWN_LOCKFILE_NAMES):
        if any(char in known for char in "*?["):
            test.assertIn(known, KNOWN_GLOB_INSTANCES, "add an instance of the glob")
            instances.append(KNOWN_GLOB_INSTANCES[known])
        else:
            instances.append(known)
    return instances


#: File names the detector must match: lockfile variants, and names it
#: matches whatever their case or whether they are known.
DETECTED_NAMES = (
    "Cargo.lock.old",
    "MODULE.bazel.lock",
    "pylock.toml",
    "pylock.dev.toml",
    "conda-lock.yml",
    "npm-shrinkwrap.json",
    "shrinkwrap.yaml",
    "packages.lock.json",
    "bun.lockb",
    "gradle.lockfile",
    "Package.resolved",
    "CARGO.LOCK",
    "lock.json",
    "fixture.lock.json",
    "build.lock",
)
#: File names the detector must not match: words ending in "lock", source
#: files named for locks, manifests and `go.sum`.
UNDETECTED_NAMES = (
    "clock.json",
    "block.json",
    "deadlock.yaml",
    "unlock.toml",
    "spin-lock.h",
    "rwlock.rs",
    "lock.rs",
    "Cargo.toml",
    "package.json",
    "go.sum",
    "lockfile.md",
)
#: The SHA-256 of the published `.crate`, `"package"` in the checksum map.
DEPENDENCY_PACKAGE_SHA256 = "c" * 64

#: The archive's two kept lockfiles, keyed by member, with the bytes the
#: fixture gives them: the project's `Cargo.lock` and the rust stub's
#: `library/Cargo.lock`.
PROJECT_LOCK = b"version = 4\n"
STD_LOCK = b"Rust lock"
KEPT_LOCK_MEMBERS = {
    "sources/Cargo.lock": PROJECT_LOCK,
    "sources/rust/source/library/Cargo.lock": STD_LOCK,
}


def cargo_checksum_json(data):
    """`.cargo-checksum.json` text as cargo writes it: compact, keys sorted,
    non-ASCII unescaped."""
    return json.dumps(data, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def tar_bytes(files):
    """An uncompressed tar of `files` (`{member name: bytes}`), byte-for-byte
    the same on every call."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


#: What the flatbuffers `git archive` stub writes: the checkout with a
#: lockfile of FlatBuffers' own, as the real tree has.
NESTED_FLATBUFFERS_LOCK = "flatbuffers/pnpm-lock.yaml"
FLATBUFFERS_TAR = tar_bytes(
    {
        "flatbuffers/LICENSE": b"Apache license",
        NESTED_FLATBUFFERS_LOCK: b"upstream FlatBuffers lock",
    }
)
#: What the rust stub writes as a standard-library dependency: a published
#: `.crate` (a gzipped tar) holding the crate's own lockfile.
NESTED_CRATE_LOCK = "libc-1/Cargo.lock"
LIBC_CRATE = gzip.compress(
    tar_bytes(
        {"libc-1/Cargo.toml": b"libc manifest", NESTED_CRATE_LOCK: b"upstream libc lock"}
    ),
    mtime=0,
)


class SourceIdentityTests(unittest.TestCase):
    def test_signed_debian_descriptor_preserves_exact_revision(self):
        text = (
            "-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA512\n\n"
            "Source: gcc-14\nVersion: 14.2.0-19\nChecksums-Sha256:\n"
            f" {'a' * 64} 42 gcc-14.orig.tar.xz\n"
            "-----BEGIN PGP SIGNATURE-----\nignored\n"
        )
        self.assertEqual(
            oss.parse_dsc(text, "gcc-14", "14.2.0-19"),
            [("gcc-14.orig.tar.xz", "a" * 64, 42)],
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            oss.parse_dsc(text, "gcc-14", "14.2.0-20")

    def test_debian_descriptor_requires_safe_checksummed_files(self):
        base = "Source: gcc-14\nVersion: 14.2.0-19\n"
        for files in (
            "",
            f"Checksums-Sha256:\n {'a' * 64} 1 ../escape\n",
            "Checksums-Sha256:\n bad 1 src.tar\n",
        ):
            with self.subTest(files=files), self.assertRaises(ValueError):
                oss.parse_dsc(base + files, "gcc-14", "14.2.0-19")

    def test_corrupt_download_is_rejected(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(oss, "urlopen", return_value=io.BytesIO(b"wrong")),
        ):
            with self.assertRaisesRegex(ValueError, "checksum"):
                oss.download(
                    "https://example.invalid/source", Path(tmp) / "source", "0" * 64
                )

    def test_truncated_download_is_rejected(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(oss, "urlopen", return_value=io.BytesIO(b"short")),
        ):
            with self.assertRaisesRegex(ValueError, "size"):
                oss.download(
                    "https://example.invalid/source", Path(tmp) / "source", size=100
                )


#: The `[patch.crates-io]` table a project that vendors a patched fred adds to
#: its root manifest, and the tree it points at.
PATCHED_TREE = "vendor/fred-10.1.0"
PATCH_TABLE = f'\n[patch.crates-io]\nfred = {{ path = "{PATCHED_TREE}" }}\n'


#: The files of the patched tree other than its lockfile, relative to it.
PATCHED_TREE_FILES = ("src/lib.rs", "LICENSE-MIT")


def vendor_patched_fred(project, records=oss.PATCHED_CRATE_RECORDS):
    """Patch fred to a vendored tree in `project`, writing the tree, with the
    published crate's `Cargo.lock`, and the sibling `records`
    (`<tree>.patch`, `<tree>.SOURCE.md`, `<tree>.SHA256SUMS`) it ships with.

    `<tree>.SHA256SUMS` is in `sha256sum` form and, like the real record,
    lists the published `Cargo.lock` the packaged tree goes without; the
    other records hold placeholder text."""
    with (project / "Cargo.toml").open("a") as manifest:
        manifest.write(PATCH_TABLE)
    tree = project / PATCHED_TREE
    (tree / "src").mkdir(parents=True)
    (tree / "src/lib.rs").write_text("patched third-party source")
    (tree / "LICENSE-MIT").write_text("fred MIT notice")
    (tree / "Cargo.lock").write_text("fred's upstream lock")
    for suffix in records:
        record = tree.with_name(tree.name + suffix)
        if suffix == ".SHA256SUMS":
            record.write_text(
                "".join(
                    f"{oss.sha256(tree / relative)}  {relative}\n"
                    for relative in sorted(("Cargo.lock", *PATCHED_TREE_FILES))
                )
            )
        else:
            record.write_text(f"fred record {suffix}")


#: The conformance image's binary (`Dockerfile.conformance`).
CONFORMANCE_BINARY = "live-edit-conformance"


@contextmanager
def packaged_fixture(prepare=None, binary=oss.SERVICE_BINARY, unlisted=()):
    """Exercise the real assembler, replacing only compiler/network boundaries.

    `prepare(project)`, when given, runs on the populated project before it is
    packaged. `binary` is the release binary the image ships as `/app/<binary>`:
    the project's build output holds only that binary, with contents of its
    own, as the image's builder leaves it. `unlisted` names files of
    DEPENDENCY_FILES the cargo stub writes but leaves out of the crate's
    `.cargo-checksum.json`.

    Every source tree carries upstream lockfiles the packaging drops: the
    vendored crate, the standard library's `stdarch` sub-project and, with
    `vendor_patched_fred`, the patched tree. The nested archives carry
    lockfiles of their own: FLATBUFFERS_TAR and LIBC_CRATE."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        project, flatbuffers = root / "project", root / "flatbuffers"
        project.mkdir()
        flatbuffers.mkdir()
        (project / "Cargo.toml").write_text(
            '[package]\nname="live-edit"\nversion="0.1.1"\n'
        )
        (project / "Cargo.lock").write_bytes(PROJECT_LOCK)
        (project / "ATTRIBUTIONS.md").write_text("Cargo notices\n")
        (project / "LICENSE.txt").write_text("Apache license\n")
        (project / "NOTICE").write_text("NVIDIA notice\n")
        (project / ".env").write_text("SECRET=must-not-be-shipped")
        (project / "src").mkdir()
        (project / "src/main.rs").write_text("proprietary source")
        (project / "target/release").mkdir(parents=True)
        (project / "target/release" / binary).write_bytes(
            f"{binary} release binary".encode()
        )
        (flatbuffers / "LICENSE").write_text("Apache license")
        (flatbuffers / "include/flatbuffers").mkdir(parents=True)
        (flatbuffers / "include/flatbuffers/flatbuffers.h").write_text(
            "/* Copyright upstream */\ncode"
        )
        if prepare is not None:
            prepare(project)

        def command(*args, **kwargs):
            if args[:2] == ("git", "describe"):
                return "v24.3.25"
            if args[:2] == ("git", "rev-parse"):
                return "abc123"
            if args[:2] == ("rustc", "--print"):
                return str(root / "sysroot")
            return "rustc 1.94.1"

        def run(args, **kwargs):
            if args[0] == "cargo":
                crate = Path(args[-1]) / DEPENDENCY_CRATE
                listed = {}
                for relative, data in DEPENDENCY_FILES.items():
                    (crate / relative).parent.mkdir(parents=True, exist_ok=True)
                    (crate / relative).write_bytes(data)
                    if relative not in unlisted:
                        listed[relative] = hashlib.sha256(data).hexdigest()
                (crate / ".cargo-checksum.json").write_text(
                    cargo_checksum_json(
                        {"files": listed, "package": DEPENDENCY_PACKAGE_SHA256}
                    ),
                    encoding="utf-8",
                )
                # jemalloc's C sources and BSD-2-Clause notice ride inside
                # the binding crate; the release binary links them statically.
                jemalloc = Path(args[-1]) / JEMALLOC_CRATE / "jemalloc"
                jemalloc.mkdir(parents=True)
                (jemalloc.parent / "Cargo.toml").write_text("binding manifest")
                (jemalloc / "COPYING").write_text("jemalloc BSD-2-Clause notice")
                (jemalloc / "src").mkdir()
                (jemalloc / "src/jemalloc.c").write_text("jemalloc source")
            else:
                filename = next(
                    arg.split("=", 1)[1] for arg in args if arg.startswith("--output=")
                )
                Path(filename).write_bytes(FLATBUFFERS_TAR)

        def rust(sysroot, destination, licenses):
            (destination / "source/library/std/src").mkdir(parents=True)
            (destination / "source/library/std/src/lib.rs").write_text("Rust source")
            (destination / "source/library/Cargo.lock").write_bytes(STD_LOCK)
            # A sub-project of the rust-src tree with a lockfile of its own.
            (destination / "source/library/stdarch").mkdir()
            (destination / "source/library/stdarch/Cargo.lock").write_text(
                "upstream stdarch lock"
            )
            (destination / "dependencies").mkdir()
            (destination / "dependencies/libc-1.crate").write_bytes(LIBC_CRATE)
            (licenses / "RUST-COPYRIGHT.html").write_text("Rust notices")
            (licenses / "rust-licenses").mkdir()
            (licenses / "rust-licenses/MIT.txt").write_text("Rust license")

        def gcc(destination, licenses):
            destination.mkdir()
            (destination / "gcc-14.dsc").write_text("source descriptor")
            (destination / "gcc-14.orig.tar.gz").write_bytes(b"GCC sources")
            (destination / "gcc-14.debian.tar.xz").write_bytes(b"Debian patches")
            for name in ("GCC-COPYRIGHT.txt", "GPL-3.txt"):
                (licenses / name).write_text("GCC license")
            return {"package": "gcc-14", "version": "14.2.0-19"}

        output = root / "image"
        with (
            patch.object(oss, "command", side_effect=command),
            patch.object(oss.subprocess, "run", side_effect=run),
            patch.object(oss, "collect_rust", side_effect=rust),
            patch.object(oss, "collect_gcc", side_effect=gcc),
        ):
            if binary == oss.SERVICE_BINARY:
                # The service's own call, unchanged: no `--binary`.
                oss.package_oss(project, flatbuffers, output, root / "work")
            else:
                oss.package_oss(project, flatbuffers, output, root / "work", binary)
        (output / "app").mkdir()
        shutil.copyfile(project / "target/release" / binary, output / "app" / binary)
        yield root, project, output


def archived_files(output):
    """`{member name: bytes}` of every regular file in the image's
    `vendor.tar.gz`; a name listed twice fails, so no member hides another."""
    files = {}
    with tarfile.open(output / "vendor.tar.gz") as archive:
        for member in archive:
            if member.isfile():
                if member.name in files:
                    raise AssertionError(f"{member.name} is archived twice")
                files[member.name] = archive.extractfile(member).read()
    return files


def nested_members(data):
    """The member names of the (possibly gzipped) tar `data`."""
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        return archive.getnames()


class UpstreamLockfileTests(unittest.TestCase):
    """Packaging drops upstream lockfiles from the archive's own files, so a
    scanner reading the archive sees only the versions the binary was built
    from; archives inside it ship as published, lockfiles and all."""

    DEPENDENCY = f"sources/crates/{DEPENDENCY_CRATE}"

    def test_the_only_archived_lockfiles_are_the_binarys_and_the_standard_librarys(self):
        for binary in (oss.SERVICE_BINARY, CONFORMANCE_BINARY):
            with (
                self.subTest(binary=binary),
                packaged_fixture(prepare=vendor_patched_fred, binary=binary) as (
                    _,
                    _,
                    output,
                ),
            ):
                files = archived_files(output)
                # Everything the detector matches is kept or was dropped.
                lockfiles = {
                    name: data
                    for name, data in files.items()
                    if oss.LOCKFILE_PATTERN.search(name.rsplit("/", 1)[-1])
                }

                self.assertEqual(lockfiles, KEPT_LOCK_MEMBERS)

    def test_a_vendored_crates_checksum_map_drops_exactly_its_removed_lockfiles(self):
        with packaged_fixture() as (_, _, output):
            files = archived_files(output)
            text = files[f"{self.DEPENDENCY}/.cargo-checksum.json"].decode("utf-8")
            expected = {
                "files": {
                    relative: hashlib.sha256(data).hexdigest()
                    for relative, data in DEPENDENCY_FILES.items()
                    if relative not in DEPENDENCY_LOCKFILES
                },
                "package": DEPENDENCY_PACKAGE_SHA256,
            }

            # Every other key and digest kept, `"package"` unchanged, written
            # in cargo's own serialization.
            self.assertEqual(text, cargo_checksum_json(expected))

    def test_a_vendored_crates_checksum_map_lists_exactly_its_archived_files(self):
        # What cargo checks when it builds from the directory: every listed
        # file is present with its digest, and nothing unlisted is there.
        with packaged_fixture() as (_, _, output):
            files = archived_files(output)
            listed = json.loads(files[f"{self.DEPENDENCY}/.cargo-checksum.json"])["files"]
            present = {
                name[len(self.DEPENDENCY) + 1 :]: hashlib.sha256(data).hexdigest()
                for name, data in files.items()
                if name.startswith(f"{self.DEPENDENCY}/")
                and name != f"{self.DEPENDENCY}/.cargo-checksum.json"
            }

            self.assertEqual(listed, present)

    def test_a_lockfile_its_crates_checksum_map_does_not_list_fails_the_packaging(self):
        for unlisted in DEPENDENCY_LOCKFILES:
            with (
                self.subTest(unlisted=unlisted),
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Vendored crate {DEPENDENCY_CRATE} does not list {unlisted} "
                        "in .cargo-checksum.json"
                    ),
                ),
                packaged_fixture(unlisted=(unlisted,)),
            ):
                pass

    def test_nested_archives_ship_unmodified_with_their_own_lockfiles(self):
        # A pin on the behaviour SOURCES.md describes: the archives inside
        # the archive ship as published, never opened, so lockfiles inside
        # them stay. A change that filters or rewrites one fails here.
        with packaged_fixture() as (_, _, output):
            files = archived_files(output)
            for name, published, lockfile in (
                ("sources/flatbuffers.tar", FLATBUFFERS_TAR, NESTED_FLATBUFFERS_LOCK),
                ("sources/rust/dependencies/libc-1.crate", LIBC_CRATE, NESTED_CRATE_LOCK),
            ):
                with self.subTest(name=name):
                    self.assertEqual(files[name], published)
                    self.assertIn(lockfile, nested_members(files[name]))


def snapshot(root):
    """`{path relative to root: bytes}` of every file under `root`."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


class LockfileClassificationTests(unittest.TestCase):
    """The detector, LOCKFILE_PATTERN, is broader than the known lockfile
    names, so a file it matches that nothing classifies stops the packaging
    instead of being deleted or shipped unchecked."""

    def test_the_detector_matches_lockfile_names_whatever_their_case(self):
        for name in DETECTED_NAMES:
            with self.subTest(name=name):
                self.assertIsNotNone(oss.LOCKFILE_PATTERN.search(name))

    def test_the_detector_does_not_match_names_that_only_resemble_a_lockfile(self):
        for name in UNDETECTED_NAMES:
            with self.subTest(name=name):
                self.assertIsNone(oss.LOCKFILE_PATTERN.search(name))

    def test_every_known_lockfile_name_is_detected(self):
        # A known name the detector missed would never be dropped.
        for name in known_lockfile_instances(self):
            with self.subTest(name=name):
                self.assertTrue(oss.is_known_lockfile(name))
                self.assertIsNotNone(oss.LOCKFILE_PATTERN.search(name))

    def test_unclassified_detections_fail_before_anything_is_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources = Path(tmp) / "sources"
            crate = sources / "crates/dep-1.0.0"
            files = {
                # The two kept, then known lockfiles the step drops when
                # nothing is unclassified.
                "Cargo.lock": b"binary lock",
                "rust/source/library/Cargo.lock": b"std lock",
                "rust/source/library/stdarch/Cargo.lock": b"stdarch lock",
                "crates/dep-1.0.0/Cargo.lock": b"dep lock",
                "patched-crates/fred-10.1.0/Cargo.lock": b"fred lock",
                # Detected, but neither known nor declared not a lockfile.
                "crates/dep-1.0.0/tests/fixture.lock.json": b"test input",
                "patched-crates/fred-10.1.0/build.lock": b"build marker",
                "rust/source/library/x/CARGO.LOCK": b"upper-case lock",
                "stray.lock": b"stray",
            }
            for relative, data in files.items():
                (sources / relative).parent.mkdir(parents=True, exist_ok=True)
                (sources / relative).write_bytes(data)
            (crate / ".cargo-checksum.json").write_text(
                cargo_checksum_json(
                    {
                        "files": {
                            key: hashlib.sha256(files[f"crates/dep-1.0.0/{key}"]).hexdigest()
                            for key in ("Cargo.lock", "tests/fixture.lock.json")
                        },
                        "package": DEPENDENCY_PACKAGE_SHA256,
                    }
                ),
                encoding="utf-8",
            )
            before = snapshot(sources)

            with self.assertRaises(ValueError) as raised:
                oss.drop_upstream_lockfiles(sources)

            message = str(raised.exception)
            # Every unclassified file, each with the tree it came from, and
            # nothing else.
            self.assertEqual(
                {line for line in message.splitlines() if line.startswith("  ")},
                {
                    "  crates/dep-1.0.0/tests/fixture.lock.json (vendored crate dep-1.0.0)",
                    "  patched-crates/fred-10.1.0/build.lock (patched crate fred-10.1.0)",
                    "  rust/source/library/x/CARGO.LOCK (the rust tree)",
                    "  stray.lock (the top of sources/)",
                },
            )
            # How to classify each: both lists, and the gate's mirror.
            for remedy in ("KNOWN_LOCKFILE_NAMES", "NOT_LOCKFILES", "not_lockfiles"):
                self.assertIn(remedy, message)
            # Nothing deleted or rewritten, the known lockfiles included.
            self.assertEqual(snapshot(sources), before)

    def test_an_upper_case_cargo_lock_fails_the_packaging(self):
        # Detected case-insensitively, known case-sensitively: `CARGO.LOCK`
        # is not a name any tool writes, so a person decides what it is.
        # (In a directory of its own, so a case-insensitive file system
        # cannot fold it into the tree's `Cargo.lock`.)
        def patched_fred_with_upper_case_lock(project):
            vendor_patched_fred(project)
            (project / PATCHED_TREE / "examples").mkdir()
            (project / PATCHED_TREE / "examples/CARGO.LOCK").write_text("lock")

        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "  patched-crates/fred-10.1.0/examples/CARGO.LOCK "
                "(patched crate fred-10.1.0)"
            ),
        ):
            with packaged_fixture(prepare=patched_fred_with_upper_case_lock):
                pass


class ArchiveTests(unittest.TestCase):
    def test_archive_contains_sources_and_identity_but_no_application_or_secrets(self):
        with packaged_fixture() as (_, project, output):
            with tarfile.open(output / "vendor.tar.gz") as archive:
                names = archive.getnames()
                self.assertIn("sources/crates/dependency-1.0.0/lib.rs", names)
                self.assertIn("sources/flatbuffers.tar", names)
                self.assertIn("sources/rust/source/library/std/src/lib.rs", names)
                self.assertIn("sources/gcc/gcc-14.dsc", names)
                self.assertFalse(
                    any(name.endswith(".env") or "main.rs" in name for name in names)
                )
                metadata = json.load(archive.extractfile("sources/manifest.json"))
                self.assertEqual(metadata["service_version"], "0.1.1")
                self.assertEqual(
                    metadata["cargo_lock_sha256"], oss.sha256(project / "Cargo.lock")
                )
                self.assertEqual(
                    metadata["binary_sha256"], oss.sha256(output / "app/live-edit")
                )

    def test_notice_ships_beside_the_license(self):
        # Apache-2.0 §4(d): NVIDIA's attribution (NOTICE) goes wherever the
        # license text goes.
        with packaged_fixture() as (_, project, output):
            for name in ("LICENSE.txt", "NOTICE"):
                with self.subTest(name=name):
                    self.assertEqual(
                        (output / "licenses" / name).read_bytes(), (project / name).read_bytes()
                    )

    def test_sources_notice_points_at_the_statically_linked_allocator_source(self):
        with packaged_fixture() as (_, _, output):
            sources = (output / "licenses/SOURCES.md").read_text()
            self.assertIn("sources/crates/tikv-jemalloc-sys-<version>/jemalloc", sources)
            self.assertIn("jemalloc/COPYING", sources)

    def test_conformance_binary_is_the_one_the_manifest_and_checksums_bind(self):
        with packaged_fixture(binary=CONFORMANCE_BINARY) as (_, _, output):
            shipped = output / "app" / CONFORMANCE_BINARY
            manifest = json.loads((output / "licenses/SOURCE-MANIFEST.json").read_text())
            self.assertEqual(manifest["binary_sha256"], oss.sha256(shipped))
            with tarfile.open(output / "vendor.tar.gz") as archive:
                self.assertEqual(
                    json.load(archive.extractfile("sources/manifest.json"))["binary_sha256"],
                    oss.sha256(shipped),
                )
            paths = [
                line.split("  ", 1)[1]
                for line in (output / "licenses/SHA256SUMS").read_text().splitlines()
            ]
            self.assertIn(f"app/{CONFORMANCE_BINARY}", paths)
            self.assertNotIn("app/live-edit", paths)
            self.assertIn(
                f"{oss.sha256(shipped)}  app/{CONFORMANCE_BINARY}",
                (output / "licenses/SHA256SUMS").read_text().splitlines(),
            )

    def test_only_the_service_image_names_the_statically_linked_allocator(self):
        # jemalloc is linked into the service binary alone, so only its
        # SOURCES.md carries the allocator paragraph.
        for binary, expected in ((oss.SERVICE_BINARY, True), (CONFORMANCE_BINARY, False)):
            with (
                self.subTest(binary=binary),
                packaged_fixture(binary=binary) as (_, _, output),
            ):
                sources = (output / "licenses/SOURCES.md").read_text()
                self.assertEqual("jemalloc" in sources, expected, sources)
                # The rest of the notice is the same for both images.
                self.assertIn("sources/patched-crates/<name>-<version>", sources)

    def test_archive_normalizes_permissions_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            file = source / "license"
            file.write_text("notice")
            file.chmod(0o600)
            oss.archive_sources(source, root / "good.tar.gz")
            with tarfile.open(root / "good.tar.gz") as archive:
                self.assertEqual(archive.getmember("sources/license").mode, 0o644)
            (source / "escape").symlink_to("/etc/passwd")
            with self.assertRaisesRegex(ValueError, "symlink"):
                oss.archive_sources(source, root / "bad.tar.gz")

    def test_unpatched_project_ships_no_patched_crates_directory(self):
        with packaged_fixture() as (_, _, output):
            with tarfile.open(output / "vendor.tar.gz") as archive:
                self.assertFalse(
                    any(name.startswith("sources/patched-crates") for name in archive.getnames())
                )

    def test_existing_output_is_not_reused(self):
        with packaged_fixture() as (root, project, output):
            with self.assertRaises(FileExistsError):
                oss.package_oss(
                    project, root / "flatbuffers", output, root / "new-work"
                )


class PatchedCrateTests(unittest.TestCase):
    """A crate `[patch.crates-io]` points at a vendored tree is compiled from
    that tree, which `cargo vendor` does not copy: `collect_patched_crates`
    ships it, with the patch and the provenance record beside it, or fails."""

    def test_patched_crate_ships_with_its_patch_and_record(self):
        with packaged_fixture(prepare=vendor_patched_fred) as (_, _, output):
            with tarfile.open(output / "vendor.tar.gz") as archive:
                names = archive.getnames()
                for name in (
                    "sources/patched-crates/fred-10.1.0/src/lib.rs",
                    "sources/patched-crates/fred-10.1.0/LICENSE-MIT",
                    "sources/patched-crates/fred-10.1.0.patch",
                    "sources/patched-crates/fred-10.1.0.SOURCE.md",
                    # The upstream-file digests SOURCE.md cites.
                    "sources/patched-crates/fred-10.1.0.SHA256SUMS",
                ):
                    with self.subTest(name=name):
                        self.assertIn(name, names)
                self.assertEqual(
                    archive.extractfile("sources/patched-crates/fred-10.1.0.patch").read(),
                    b"fred record .patch",
                )

    def test_patched_crate_ships_without_its_lockfile_and_with_its_records_unchanged(self):
        with packaged_fixture(prepare=vendor_patched_fred) as (_, project, output):
            files = archived_files(output)
            tree = "sources/patched-crates/fred-10.1.0"

            self.assertNotIn(f"{tree}/Cargo.lock", files)
            for relative in PATCHED_TREE_FILES:
                with self.subTest(path=relative):
                    self.assertEqual(
                        files[f"{tree}/{relative}"],
                        (project / PATCHED_TREE / relative).read_bytes(),
                    )
            # The records ship as committed. `.SHA256SUMS` lists the published
            # crate's `Cargo.lock`, since it records that release, so dropping
            # the lockfile must not drop its line there.
            self.assertIn(
                "  Cargo.lock\n",
                (project / f"{PATCHED_TREE}.SHA256SUMS").read_text(),
                "the fixture record must list the lockfile for this to discriminate",
            )
            for suffix in oss.PATCHED_CRATE_RECORDS:
                with self.subTest(record=suffix):
                    self.assertEqual(
                        files[f"{tree}{suffix}"],
                        (project / f"{PATCHED_TREE}{suffix}").read_bytes(),
                    )

    def test_a_missing_patch_or_provenance_record_fails_the_packaging(self):
        for absent in (".patch", ".SOURCE.md", ".SHA256SUMS"):
            present = tuple(s for s in (".patch", ".SOURCE.md", ".SHA256SUMS") if s != absent)
            with self.subTest(absent=absent):
                with self.assertRaisesRegex(
                    ValueError, f"Patched crate record is missing: fred-10.1.0{absent}"
                ):
                    with packaged_fixture(
                        prepare=lambda project: vendor_patched_fred(project, records=present)
                    ):
                        pass

    def test_a_missing_patched_tree_fails_the_packaging(self):
        def patch_without_a_tree(project):
            with (project / "Cargo.toml").open("a") as manifest:
                manifest.write(PATCH_TABLE)

        with self.assertRaisesRegex(
            ValueError, f"Patched crate tree is missing: {PATCHED_TREE}"
        ):
            with packaged_fixture(prepare=patch_without_a_tree):
                pass

    def test_a_registry_patch_without_a_path_ships_nothing(self):
        # Only a `path` patch names a local tree; a git or registry patch is
        # fetched by `cargo vendor` like any other package.
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "patched-crates"
            oss.collect_patched_crates(
                Path(tmp),
                {"patch": {"crates-io": {"fred": {"git": "https://example.invalid/fred"}}}},
                destination,
            )
            self.assertFalse(destination.exists())


def rewrite_archive(root, output, keep=lambda name: True, extra=None):
    """Rewrite the image's `vendor.tar.gz` with only the members whose name
    `keep` accepts, then the `extra` files (`{member name: bytes}`) appended."""
    replacement = root / "replacement.tar.gz"
    with (
        tarfile.open(output / "vendor.tar.gz") as original,
        tarfile.open(replacement, "w:gz") as updated,
    ):
        for member in original:
            if keep(member.name):
                updated.addfile(
                    member, original.extractfile(member) if member.isfile() else None
                )
        for name, data in (extra or {}).items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            updated.addfile(info, io.BytesIO(data))
    replacement.replace(output / "vendor.tar.gz")


def rewrite_checksums(output):
    """Rewrite `licenses/SHA256SUMS` to match the image as it now is, so a
    test isolates the check it targets from the independent checksum-integrity
    check."""
    checksums = output / "licenses/SHA256SUMS"
    files = sorted(
        path for path in output.rglob("*") if path.is_file() and path != checksums
    )
    checksums.write_text(
        "".join(f"{oss.sha256(path)}  {path.relative_to(output)}\n" for path in files)
    )


class ContainerGateTests(unittest.TestCase):
    def run_gate(self, root, project, output, *binary, gate=None):
        """Runs the gate on the image laid out in `output`; `binary`, when
        given, is the gate's BINARY argument. `gate` is the script to run,
        by default the repository's."""
        fake_bin = root / "bin"
        fake_bin.mkdir(exist_ok=True)
        docker = fake_bin / "docker"
        docker.write_text("""#!/usr/bin/env python3
import os, pathlib, shutil, sys
if sys.argv[1] == "create":
    print("test-container")
elif sys.argv[1] == "cp":
    source = pathlib.Path(os.environ["TEST_IMAGE_ROOT"]) / sys.argv[2].split(":/", 1)[1]
    destination = pathlib.Path(sys.argv[3])
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copyfile(source, destination)
elif sys.argv[1] != "rm":
    raise SystemExit(1)
""")
        docker.chmod(0o755)
        return subprocess.run(
            [
                "sh",
                str(gate or GATE),
                "release@sha256:example",
                *binary,
            ],
            cwd=project,
            env={
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "TEST_IMAGE_ROOT": str(output),
            },
            capture_output=True,
            text=True,
        )

    def test_complete_image_passes(self):
        # With and without a patched crate: every tree's upstream lockfiles
        # are dropped, and the nested archives' own lockfiles are not counted.
        for prepare in (None, vendor_patched_fred):
            with (
                self.subTest(patched=prepare is not None),
                packaged_fixture(prepare=prepare) as (root, project, output),
            ):
                result = self.run_gate(root, project, output)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_notices_fail(self):
        for name in ("ATTRIBUTIONS.md", "LICENSE.txt", "NOTICE"):
            with self.subTest(name=name), packaged_fixture() as (root, project, output):
                (output / "licenses" / name).unlink()
                self.assertNotEqual(self.run_gate(root, project, output).returncode, 0)

    def test_incomplete_or_duplicated_checksum_inventory_fails(self):
        for duplicate in (False, True):
            with (
                self.subTest(duplicate=duplicate),
                packaged_fixture() as (root, project, output),
            ):
                checksums = output / "licenses/SHA256SUMS"
                lines = checksums.read_text().splitlines(keepends=True)
                checksums.write_text(
                    "".join(lines + lines[:1])
                    if duplicate
                    else "".join(
                        line for line in lines if not line.endswith("  app/live-edit\n")
                    )
                )
                self.assertNotEqual(self.run_gate(root, project, output).returncode, 0)

    def test_missing_native_sources_or_license_category_fails_with_valid_checksums(
        self,
    ):
        # `jemalloc/COPYING`: the statically linked allocator's notice, dropped
        # alone while the rest of its vendored crate stays in the archive.
        for category in (
            "rust-licenses",
            "orig.tar.gz",
            "debian.tar.xz",
            "jemalloc/COPYING",
        ):
            with (
                self.subTest(category=category),
                packaged_fixture() as (root, project, output),
            ):
                if category == "rust-licenses":
                    shutil.rmtree(output / "licenses/rust-licenses")
                else:
                    rewrite_archive(
                        root, output, keep=lambda name: not name.endswith(category)
                    )
                rewrite_checksums(output)
                self.assertNotEqual(self.run_gate(root, project, output).returncode, 0)

    def assert_gate_failed(self, result, line=None):
        """The gate failed an image check (exit 1, not the usage refusal's 2)
        and, when `line` is given, printed it to stderr as a line of its own."""
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        if line is not None:
            self.assertIn(line, result.stderr.splitlines(), result.stderr)

    @staticmethod
    def lockfile_problem(kind, member):
        """The gate's stderr line for a lockfile member of `kind`
        (`unexpected`, `missing` or `duplicated`)."""
        return f"check-container-oss.sh: {kind} lockfile member: {member}"

    def test_an_image_rewritten_unchanged_passes(self):
        # Control for every failure test built on the two helpers: rewriting
        # the archive and the checksums alone leaves the image passing, so
        # each of those tests fails on the change it makes.
        with packaged_fixture() as (root, project, output):
            rewrite_archive(root, output)
            rewrite_checksums(output)
            result = self.run_gate(root, project, output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_an_upstream_lockfile_in_the_archive_fails_with_valid_checksums(self):
        # An instance of every known lockfile name, and detected names that
        # no list classifies: the gate rejects every detected member but the
        # kept ones, known or not.
        names = sorted(set(known_lockfile_instances(self)) | set(DETECTED_NAMES))
        for name in names:
            member = f"sources/crates/{DEPENDENCY_CRATE}/{name}"
            with (
                self.subTest(name=name),
                packaged_fixture() as (root, project, output),
            ):
                rewrite_archive(root, output, extra={member: b"upstream lock"})
                rewrite_checksums(output)
                self.assert_gate_failed(
                    self.run_gate(root, project, output),
                    self.lockfile_problem("unexpected", member),
                )

    def test_the_gate_detects_exactly_the_members_the_packager_detects(self):
        # The same pattern text, run by awk on the lower-cased name in the
        # gate and by Python's `re`, case-insensitively, in the packager:
        # the gate reports exactly the injected members the packager's
        # detector matches.
        names = (
            *DETECTED_NAMES,
            *UNDETECTED_NAMES,
            *known_lockfile_instances(self),
            "pylock.a.b.toml",
            "Package.Resolved",
            "x.LOCKFILE",
            "Lock.TOML",
            "foo_lock.yaml",
            "pnpm-lock.yml",
            "a.lock.b",
            "lock",
            "pylock.tomlx",
        )
        members = sorted({f"sources/crates/{DEPENDENCY_CRATE}/x/{name}" for name in names})
        with packaged_fixture() as (root, project, output):
            rewrite_archive(root, output, extra={member: b"" for member in members})
            rewrite_checksums(output)
            result = self.run_gate(root, project, output)

            self.assertEqual(
                [
                    line
                    for line in result.stderr.splitlines()
                    if " lockfile member: " in line
                ],
                [
                    self.lockfile_problem("unexpected", member)
                    for member in members
                    if oss.LOCKFILE_PATTERN.search(member.rsplit("/", 1)[-1])
                ],
            )

    def test_a_missing_or_duplicated_kept_lockfile_fails_with_valid_checksums(self):
        for member in sorted(KEPT_LOCK_MEMBERS):
            for duplicate in (False, True):
                with (
                    self.subTest(member=member, duplicate=duplicate),
                    packaged_fixture() as (root, project, output),
                ):
                    if duplicate:
                        # A second member of the same name: the archive then
                        # lists that lockfile twice.
                        rewrite_archive(
                            root, output, extra={member: KEPT_LOCK_MEMBERS[member]}
                        )
                    else:
                        rewrite_archive(root, output, keep=lambda name: name != member)
                    rewrite_checksums(output)
                    self.assert_gate_failed(
                        self.run_gate(root, project, output),
                        self.lockfile_problem(
                            "duplicated" if duplicate else "missing", member
                        ),
                    )

    def test_a_not_lockfile_entry_ships_and_passes_the_gate_that_mirrors_it(self):
        # A detected file declared not a lockfile, in NOT_LOCKFILES and, under
        # `sources/`, in the gate's not_lockfiles. Both lists are empty in the
        # repository, so each side is given the entry here.
        relative = "patched-crates/fred-10.1.0/tests/fixture.lock.json"
        member = f"sources/{relative}"

        def patched_fred_with_fixture(project):
            vendor_patched_fred(project)
            (project / PATCHED_TREE / "tests").mkdir()
            (project / PATCHED_TREE / "tests/fixture.lock.json").write_text("{}")

        with (
            patch.object(oss, "NOT_LOCKFILES", frozenset({relative})),
            packaged_fixture(prepare=patched_fred_with_fixture) as (root, project, output),
        ):
            self.assertIn(member, archived_files(output))
            script = GATE.read_text()
            self.assertIn("\nnot_lockfiles=''\n", script)
            mirrored = root / "check-container-oss.sh"
            mirrored.write_text(
                script.replace("\nnot_lockfiles=''\n", f"\nnot_lockfiles='{member}'\n")
            )

            passed = self.run_gate(root, project, output, gate=mirrored)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
            # Without the mirror the gate rejects the member.
            self.assert_gate_failed(
                self.run_gate(root, project, output),
                self.lockfile_problem("unexpected", member),
            )

    def test_the_gates_lockfile_lists_are_the_packagers(self):
        # The gate is shell and cannot import the packager's constants; its
        # copies must be the same, or a member one side drops or ships the
        # other rejects or misses.
        script = GATE.read_text()

        def assigned(variable):
            found = re.search(rf"^{variable}='([^']*)'$", script, re.MULTILINE)
            self.assertIsNotNone(found, f"the gate no longer assigns {variable}")
            return found.group(1)

        pattern = oss.LOCKFILE_PATTERN.pattern
        self.assertEqual(assigned("lockfile_pattern"), pattern)
        # The gate matches the lower-cased name, so the shared text must be
        # lower case and the packager's match case-insensitive.
        self.assertEqual(pattern, pattern.lower())
        self.assertTrue(oss.LOCKFILE_PATTERN.flags & re.IGNORECASE)
        for variable, constant in (
            ("kept_lockfiles", oss.KEPT_LOCKFILES),
            ("not_lockfiles", oss.NOT_LOCKFILES),
        ):
            with self.subTest(variable=variable):
                self.assertEqual(
                    sorted(assigned(variable).split()),
                    sorted(f"sources/{path}" for path in constant),
                )

    def test_complete_conformance_image_passes_with_its_binary(self):
        with packaged_fixture(binary=CONFORMANCE_BINARY) as (root, project, output):
            result = self.run_gate(root, project, output, CONFORMANCE_BINARY)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_conformance_image_checked_as_the_service_image_fails(self):
        # The default BINARY is the service's: the conformance image has no
        # /app/live-edit.
        with packaged_fixture(binary=CONFORMANCE_BINARY) as (root, project, output):
            self.assertNotEqual(self.run_gate(root, project, output).returncode, 0)

    def test_conformance_image_whose_checksums_name_the_service_binary_fails(self):
        # The image carries the right binary, but SHA256SUMS binds
        # `app/live-edit` (its hash kept), as a packaging run without
        # `--binary` would leave it.
        with packaged_fixture(binary=CONFORMANCE_BINARY) as (root, project, output):
            checksums = output / "licenses/SHA256SUMS"
            text = checksums.read_text()
            self.assertIn(f"  app/{CONFORMANCE_BINARY}\n", text)
            checksums.write_text(
                text.replace(f"  app/{CONFORMANCE_BINARY}\n", "  app/live-edit\n")
            )
            self.assertNotEqual(
                self.run_gate(root, project, output, CONFORMANCE_BINARY).returncode, 0
            )

    def test_binary_that_is_not_a_file_name_is_refused(self):
        for binary in (f"sub/{CONFORMANCE_BINARY}", f"../app/{CONFORMANCE_BINARY}", ""):
            with (
                self.subTest(binary=binary),
                packaged_fixture(binary=CONFORMANCE_BINARY) as (root, project, output),
            ):
                result = self.run_gate(root, project, output, binary)
                # Exit 2 is the refusal; every check of the image fails 1.
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("BINARY must be a file name", result.stderr)

    def test_mismatched_binary_and_corrupt_source_archive_fail(self):
        for name in ("app/live-edit", "vendor.tar.gz"):
            with self.subTest(name=name), packaged_fixture() as (root, project, output):
                (output / name).write_bytes(b"wrong artifact")
                self.assertNotEqual(self.run_gate(root, project, output).returncode, 0)


if __name__ == "__main__":
    unittest.main()
