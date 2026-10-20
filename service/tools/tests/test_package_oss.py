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
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools import package_oss as oss


ROOT = Path(__file__).resolve().parents[2]
# The vendored directory `cargo vendor --versioned-dirs` writes for the locked
# jemalloc binding. The real name, `+` build metadata included, so the gate's
# member pattern is exercised against what the release build actually vendors.
JEMALLOC_CRATE = (
    "tikv-jemalloc-sys-0.6.1+5.3.0-1-ge13ca993e8ccb9ba9847cc330696e02839f328f7"
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


def vendor_patched_fred(project, records=oss.PATCHED_CRATE_RECORDS):
    """Patch fred to a vendored tree in `project`, writing the tree and the
    sibling `records` (`<tree>.patch`, `<tree>.SOURCE.md`,
    `<tree>.SHA256SUMS`) it ships with."""
    with (project / "Cargo.toml").open("a") as manifest:
        manifest.write(PATCH_TABLE)
    tree = project / PATCHED_TREE
    (tree / "src").mkdir(parents=True)
    (tree / "src/lib.rs").write_text("patched third-party source")
    (tree / "LICENSE-MIT").write_text("fred MIT notice")
    for suffix in records:
        tree.with_name(tree.name + suffix).write_text(f"fred record {suffix}")


@contextmanager
def packaged_fixture(prepare=None):
    """Exercise the real assembler, replacing only compiler/network boundaries.

    `prepare(project)`, when given, runs on the populated project before it is
    packaged."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        project, flatbuffers = root / "project", root / "flatbuffers"
        project.mkdir()
        flatbuffers.mkdir()
        (project / "Cargo.toml").write_text(
            '[package]\nname="live-edit"\nversion="0.1.1"\n'
        )
        (project / "Cargo.lock").write_text("version = 4\n")
        (project / "ATTRIBUTIONS.md").write_text("Cargo notices\n")
        (project / "LICENSE.txt").write_text("Apache license\n")
        (project / "NOTICE").write_text("NVIDIA notice\n")
        (project / ".env").write_text("SECRET=must-not-be-shipped")
        (project / "src").mkdir()
        (project / "src/main.rs").write_text("proprietary source")
        (project / "target/release").mkdir(parents=True)
        (project / "target/release/live-edit").write_bytes(b"release binary")
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
                crate = Path(args[-1]) / "dependency-1.0.0"
                crate.mkdir(parents=True)
                (crate / "Cargo.toml").write_text("third-party manifest")
                (crate / "lib.rs").write_text("third-party source")
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
                with tarfile.open(filename, "w") as archive:
                    archive.add(flatbuffers / "LICENSE", arcname="flatbuffers/LICENSE")

        def rust(sysroot, destination, licenses):
            (destination / "source/library/std/src").mkdir(parents=True)
            (destination / "source/library/std/src/lib.rs").write_text("Rust source")
            (destination / "source/library/Cargo.lock").write_text("Rust lock")
            (destination / "dependencies").mkdir()
            (destination / "dependencies/libc-1.crate").write_bytes(b"source archive")
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
            oss.package_oss(project, flatbuffers, output, root / "work")
        (output / "app").mkdir()
        shutil.copyfile(project / "target/release/live-edit", output / "app/live-edit")
        yield root, project, output


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


class ContainerGateTests(unittest.TestCase):
    def run_gate(self, root, project, output):
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
                str(ROOT / "tools/check-container-oss.sh"),
                "release@sha256:example",
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
        with packaged_fixture() as (root, project, output):
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
                    replacement = root / "replacement.tar.gz"
                    with (
                        tarfile.open(output / "vendor.tar.gz") as original,
                        tarfile.open(replacement, "w:gz") as updated,
                    ):
                        for member in original:
                            if not member.name.endswith(category):
                                updated.addfile(
                                    member,
                                    original.extractfile(member)
                                    if member.isfile()
                                    else None,
                                )
                    replacement.replace(output / "vendor.tar.gz")
                # Keep hashes valid to isolate category completeness from the
                # independent checksum-integrity check.
                checksums = output / "licenses/SHA256SUMS"
                files = sorted(
                    path
                    for path in output.rglob("*")
                    if path.is_file() and path != checksums
                )
                checksums.write_text(
                    "".join(
                        f"{oss.sha256(path)}  {path.relative_to(output)}\n"
                        for path in files
                    )
                )
                self.assertNotEqual(self.run_gate(root, project, output).returncode, 0)

    def test_mismatched_binary_and_corrupt_source_archive_fail(self):
        for name in ("app/live-edit", "vendor.tar.gz"):
            with self.subTest(name=name), packaged_fixture() as (root, project, output):
                (output / name).write_bytes(b"wrong artifact")
                self.assertNotEqual(self.run_gate(root, project, output).returncode, 0)


if __name__ == "__main__":
    unittest.main()
