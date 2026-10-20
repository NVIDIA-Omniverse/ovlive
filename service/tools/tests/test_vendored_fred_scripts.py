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

"""Tests for the vendored-fred CI scripts, `scripts/check-vendored-fred.sh`
and `scripts/audit-vendored-fred.sh`, run as subprocesses under `sh`, and
for the committed record they check against.

Both scripts locate the repository as the parent of their own directory, so
each test builds a throwaway repository in a temp dir and symlinks the real
script into its `scripts/`: the bytes under test are the committed script,
and every file it reads is the test's own fixture.

`CheckVendoredFredTests` covers the drift check's last step, which requires
the `fred` path entry to sit in the root `Cargo.toml`'s `[patch.crates-io]`
table. Each case is the repository's real `Cargo.toml` with that table
rewritten. The earlier steps are preconditions here, satisfied by a
synthetic fixture: a local `.crate` (`FRED_CRATE_ARCHIVE`) whose checksum
the record carries, a vendored tree identical to it, and a stub `patch` on
`PATH`, so the tests need neither the network nor GNU patch. A stub `curl`
fails the run if the script ever tries the network. Every failing case also
asserts the step-6 message, so it cannot pass by failing an earlier step.

`CheckVendoredFredTests` also covers the release fence that follows: the
root manifest names fred's `live-edit-retention-report` feature once, on its
`test-support` line, and `cargo tree --locked` resolves the feature only with
`live-edit/test-support`. A stub `cargo` on `PATH` prints a canned
`cargo tree` per call (the release resolution, or the one with
`live-edit/test-support`), so the tests need no Rust toolchain; whether the
real `cargo tree` still prints the node the fence greps for is the fence's
own control step, which CI's `fred:vendor-drift-check` job runs with cargo.
The same stub answers the replicas fence's `cargo tree --all-features` call
(`CARGO_TREE_ALL`): it passes when that resolution has no fred `replicas`
node, and fails when it has one or when the call cannot run.

`AuditVendoredFredTests` covers what the audit script does around
`cargo audit`, with a stub `cargo` on `PATH` that prints a canned report and
exits with a chosen status: finding the source-less fred in `Cargo.lock`,
the one-package lockfile it audits, and telling an advisory against fred
from an advisory against cargo-audit itself and from an audit that could not
run. Whether the real cargo-audit still prints the lines the script greps
for is not covered here; `.gitlab-ci.yml` pins `CARGO_AUDIT_VERSION` for
that reason.

`VendoredTreeRecordTests` reads the committed tree, patch and upstream-file
manifest (`vendor/fred-<version>.SHA256SUMS`) directly, with no script and no
network, through the parsers `tools/license_headers.py` uses for the same
record: the manifest lists every file of the published crate, the files the
patch leaves alone still carry their listed digests, the files it edits do
not, the files it adds are not listed, those added files, NVIDIA's own,
start with NVIDIA's Apache-2.0 header, and the license check passes every
file of the tree.

The scripts use `awk`; the CI image's is mawk. These tests are stdlib
`unittest`, matching the `supply-chain:python-tests` job. Run from the repo
root with:

    python3 -m unittest tools.tests.test_vendored_fred_scripts
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import unittest
from pathlib import Path

from tools import license_headers as lh


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check-vendored-fred.sh"
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "audit-vendored-fred.sh"
CARGO_TOML = REPO_ROOT / "Cargo.toml"

CRATE = "fred"


def _check_script_version():
    """The fred version `check-vendored-fred.sh` checks (its `VERSION=`)."""
    match = re.search(r"^VERSION=(\S+)$", CHECK_SCRIPT.read_text(encoding="utf-8"), re.M)
    if match is None:
        raise AssertionError("no `VERSION=` line in {}".format(CHECK_SCRIPT))
    return match.group(1)


VERSION = _check_script_version()
TREE = "vendor/{}-{}".format(CRATE, VERSION)

#: The root manifest's `[patch.crates-io]` table as committed, which every
#: `CheckVendoredFredTests` case rewrites.
PATCH_HEADER = "[patch.crates-io]"
PATCH_ENTRY = '{} = {{ path = "{}" }}'.format(CRATE, TREE)
PATCH_TABLE = "{}\n{}\n".format(PATCH_HEADER, PATCH_ENTRY)

REGISTRY_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"

#: Bound on one script run. A healthy run takes well under a second; this
#: only turns a hang into a failure.
RUN_TIMEOUT_SECS = 60


@unittest.skipUnless(
    os.name == "posix" and shutil.which("sh") and shutil.which("awk"),
    "the scripts under test are POSIX sh and need sh and awk",
)
class _ScriptHarness(unittest.TestCase):
    """A temp repository with the script under test symlinked into its
    `scripts/`, and a stub directory at the front of `PATH`."""

    SCRIPT: Path

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmpdir = Path(tmp.name)
        self.repo = self.tmpdir / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "vendor").mkdir()
        os.symlink(self.SCRIPT, self.repo / "scripts" / self.SCRIPT.name)
        self.bin = self.tmpdir / "bin"
        self.bin.mkdir()
        scratch = self.tmpdir / "scratch"
        scratch.mkdir()
        self.env = dict(os.environ)
        # An inherited `DRIFT_WORK_DIR` would make the drift check reuse and
        # keep a directory outside this test's temp tree.
        self.env.pop("DRIFT_WORK_DIR", None)
        self.env["PATH"] = "{}{}{}".format(self.bin, os.pathsep, os.environ.get("PATH", ""))
        # The scripts' `mktemp -d` work dirs land here, not in the shared /tmp.
        self.env["TMPDIR"] = str(scratch)
        self.write_stub(
            "curl",
            'echo "curl stub: this test must not reach the network" >&2\nexit 97\n',
        )

    def write_stub(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        path.chmod(0o755)

    def write_record(self, version, checksum, patch_text=""):
        """`vendor/fred-<version>.SOURCE.md` with the checksum line both
        scripts parse and the `diff` block the drift check compares."""
        record = self.repo / "vendor" / "{}-{}.SOURCE.md".format(CRATE, version)
        record.write_text(
            "# fred {v}\n\n**Crate SHA256:** `{c}`\n\n```diff\n{p}```\n".format(
                v=version, c=checksum, p=patch_text
            ),
            encoding="utf-8",
        )

    def run_script(self, *args):
        """Run the symlinked script; returns `(exit code, stdout + stderr)`."""
        proc = subprocess.run(
            ["sh", str(self.repo / "scripts" / self.SCRIPT.name), *args],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=RUN_TIMEOUT_SECS,
        )
        return proc.returncode, proc.stdout


# ── check-vendored-fred.sh: the [patch.crates-io] entry ───────────────────


class CheckVendoredFredTests(_ScriptHarness):
    """The fred path entry counts only inside `[patch.crates-io]`: the same
    line under any other table is a path dependency or nothing, and does
    not replace the registry fred the dependency graph pulls in."""

    SCRIPT = CHECK_SCRIPT

    #: The step-6 failure, as the script prints it.
    NOT_PATCHED = "Cargo.toml does not patch {} to {} under [patch.crates-io]".format(
        CRATE, TREE
    )
    PASSED = "OK: {} is the published {} {} crate".format(TREE, CRATE, VERSION)

    #: The fence's feature, and the node `cargo tree -e features` prints for it.
    RETENTION_FEATURE = "live-edit-retention-report"
    RESOLVED_NODE = '{} feature "{}"'.format(CRATE, RETENTION_FEATURE)
    #: A healthy resolution: the feature appears only with test-support.
    RELEASE_TREE = "{} v{} (/repo/{})".format(CRATE, VERSION, TREE)
    TEST_TREE = "\n".join(
        (RELEASE_TREE, RESOLVED_NODE, '\u2514\u2500\u2500 live-edit feature "test-support"')
    )
    FENCE_PASSED = "OK: {}'s {} feature is enabled only by live-edit's test-support".format(
        CRATE, RETENTION_FEATURE
    )
    #: The replicas fence's feature, the node it greps for, and its pass line.
    REPLICAS_NODE = '{} feature "replicas"'.format(CRATE)
    REPLICAS_PASSED = "OK: {}'s replicas feature is off in every workspace build.".format(CRATE)

    #: The synthetic crate: a manifest, a source file, and the two files the
    #: GitHub source release drops (`PUBLISH_DROPPED` in the script).
    CRATE_FILES = {
        "Cargo.toml": '[package]\nname = "fred"\nversion = "{}"\n'.format(VERSION),
        "src/lib.rs": "pub fn stand_in() {}\n",
        "Cargo.lock": "version = 3\n",
        ".vscode/settings.json": "{}\n",
    }
    DIFFERS = "{} differs from the published crate".format(TREE)
    PATCH_TEXT = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-x\n+x\n"

    def setUp(self):
        super().setUp()
        archive = self.tmpdir / "{}-{}.crate".format(CRATE, VERSION)
        with tarfile.open(archive, "w:gz") as tf:
            for relative, text in self.CRATE_FILES.items():
                payload = text.encode("utf-8")
                info = tarfile.TarInfo("{}-{}/{}".format(CRATE, VERSION, relative))
                info.size = len(payload)
                info.mtime = 0
                tf.addfile(info, io.BytesIO(payload))
        tree = self.repo / TREE
        for relative, text in self.CRATE_FILES.items():
            (tree / relative).parent.mkdir(parents=True, exist_ok=True)
            (tree / relative).write_text(text, encoding="utf-8")
        (self.repo / (TREE + ".patch")).write_text(self.PATCH_TEXT, encoding="utf-8")
        self.write_record(
            VERSION, hashlib.sha256(archive.read_bytes()).hexdigest(), self.PATCH_TEXT
        )
        # The fences' three `cargo tree` calls, told apart by whether the call
        # selects every feature (`--all-features`, the replicas fence) or
        # `live-edit/test-support` (the retention fence's control): each prints
        # its canned resolution (`CARGO_TREE_ALL` / `CARGO_TREE_TEST` /
        # `CARGO_TREE_RELEASE`) and exits with its own status
        # (`CARGO_TREE_ALL_EXIT` / `CARGO_TREE_TEST_EXIT` /
        # `CARGO_TREE_RELEASE_EXIT`, default 0).
        self.write_stub(
            "cargo",
            'case " $* " in\n'
            '  *" --all-features "*)\n'
            '    printf \'%s\\n\' "$CARGO_TREE_ALL"; exit "${CARGO_TREE_ALL_EXIT:-0}" ;;\n'
            '  *" live-edit/test-support "*)\n'
            '    printf \'%s\\n\' "$CARGO_TREE_TEST"; exit "${CARGO_TREE_TEST_EXIT:-0}" ;;\n'
            '  *)\n'
            '    printf \'%s\\n\' "$CARGO_TREE_RELEASE"; exit "${CARGO_TREE_RELEASE_EXIT:-0}" ;;\n'
            "esac\n",
        )
        self.env["CARGO_TREE_RELEASE"] = self.RELEASE_TREE
        self.env["CARGO_TREE_TEST"] = self.TEST_TREE
        self.env["CARGO_TREE_ALL"] = self.TEST_TREE
        self.env["FRED_CRATE_ARCHIVE"] = str(archive)
        # The tree is the crate unchanged, so applying the patch is a no-op.
        self.write_stub("patch", "exit 0\n")
        self.real_manifest = CARGO_TOML.read_text(encoding="utf-8")

    def run_with_patch_table(self, table):
        """Run against the real `Cargo.toml` with its `[patch.crates-io]`
        table replaced by `table`."""
        self.assertEqual(
            self.real_manifest.count(PATCH_TABLE),
            1,
            "{} no longer carries the table these cases rewrite:\n{}".format(
                CARGO_TOML, PATCH_TABLE
            ),
        )
        (self.repo / "Cargo.toml").write_text(
            self.real_manifest.replace(PATCH_TABLE, table), encoding="utf-8"
        )
        return self.run_script()

    def run_with_real_patch_table(self):
        (self.repo / "Cargo.toml").write_text(self.real_manifest, encoding="utf-8")
        return self.run_script()

    def assert_absence_passes(self, *gone):
        """The GitHub copy lacks `Cargo.lock` and `.vscode/settings.json`
        (git ignore rules drop them when the release is committed): their
        absence passes and is noted."""
        for rel in gone:
            (self.repo / TREE / rel).unlink()
        if ".vscode/settings.json" in gone:
            (self.repo / TREE / ".vscode").rmdir()
        code, out = self.run_with_real_patch_table()
        self.assertEqual(code, 0, out)
        self.assertIn(self.PASSED, out)
        for rel in gone:
            self.assertIn("note: {}/{} is absent; accepted".format(TREE, rel), out)

    def test_an_absent_cargo_lock_passes(self):
        self.assert_absence_passes("Cargo.lock")

    def test_an_absent_vscode_settings_passes(self):
        self.assert_absence_passes(".vscode/settings.json")

    def test_both_publish_dropped_files_absent_passes(self):
        self.assert_absence_passes("Cargo.lock", ".vscode/settings.json")

    def test_a_publish_dropped_file_that_is_present_must_match(self):
        (self.repo / TREE / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")
        code, out = self.run_with_real_patch_table()
        self.assertNotEqual(code, 0, out)
        self.assertIn(self.DIFFERS, out)

    def test_a_differing_present_vscode_settings_fails(self):
        (self.repo / TREE / ".vscode/settings.json").write_text('{"x": 1}\n', encoding="utf-8")
        code, out = self.run_with_real_patch_table()
        self.assertNotEqual(code, 0, out)
        self.assertIn(self.DIFFERS, out)

    def test_no_note_when_both_publish_dropped_files_are_present(self):
        code, out = self.run_with_real_patch_table()
        self.assertEqual(code, 0, out)
        self.assertNotIn("note:", out)

    def test_this_repository_requires_both_publish_dropped_files(self):
        """The absence is accepted only in the published copy, which has no
        `.gitlab-ci.yml`: here a re-vendor that forgot the ignored `Cargo.lock`
        still fails."""
        (self.repo / ".gitlab-ci.yml").write_text("stages: []\n", encoding="utf-8")
        for rel in ("Cargo.lock", ".vscode/settings.json"):
            with self.subTest(gone=rel):
                path = self.repo / TREE / rel
                original = path.read_text(encoding="utf-8")
                path.unlink()
                try:
                    code, out = self.run_with_real_patch_table()
                finally:
                    path.write_text(original, encoding="utf-8")
                self.assertNotEqual(code, 0, out)
                self.assertIn(self.DIFFERS, out)
                self.assertNotIn("note:", out)

    def test_any_other_absent_file_fails(self):
        (self.repo / TREE / "src/lib.rs").unlink()
        code, out = self.run_with_real_patch_table()
        self.assertNotEqual(code, 0, out)
        self.assertIn(self.DIFFERS, out)

    def test_patch_entry_inside_the_patch_table_passes(self):
        for case, table in (
            ("the committed table", PATCH_TABLE),
            ("spaced header", "[ patch . crates-io ]\n{}\n".format(PATCH_ENTRY)),
            ("another table after it", PATCH_TABLE + "\n[profile.dev]\ndebug = 1\n"),
        ):
            with self.subTest(case=case):
                code, output = self.run_with_patch_table(table)

                self.assertEqual(code, 0, output)
                self.assertIn(self.PASSED, output)

    def test_patch_entry_outside_the_patch_table_fails(self):
        for case, table in (
            (
                "quoted target table after the patch table",
                "{}\n[target.'cfg(unix)'.dependencies]\n{}\n".format(PATCH_HEADER, PATCH_ENTRY),
            ),
            (
                "spaced dependencies header",
                "{}\n[ dependencies ]\n{}\n".format(PATCH_HEADER, PATCH_ENTRY),
            ),
            ("array-of-tables header", "{}\n[[bin]]\n{}\n".format(PATCH_HEADER, PATCH_ENTRY)),
            ("entry above the header", "{}\n{}\n".format(PATCH_ENTRY, PATCH_HEADER)),
            ("entry commented out", "{}\n# {}\n".format(PATCH_HEADER, PATCH_ENTRY)),
            ("quoted patch header", '[patch."crates-io"]\n{}\n'.format(PATCH_ENTRY)),
            # The dots in the tree path are literal: `.` unescaped in the
            # entry regex would match any character.
            (
                "repointed to a look-alike path",
                '{}\n{} = {{ path = "vendor/{}-{}" }}\n'.format(
                    PATCH_HEADER, CRATE, CRATE, VERSION.replace(".", "x")
                ),
            ),
            (
                "repointed to another version",
                '{}\n{} = {{ path = "vendor/{}-9.9.9" }}\n'.format(PATCH_HEADER, CRATE, CRATE),
            ),
        ):
            with self.subTest(case=case):
                code, output = self.run_with_patch_table(table)

                self.assertEqual(code, 1, output)
                self.assertIn(self.NOT_PATCHED, output)

    def run_committed(self):
        """Run against the real `Cargo.toml` unchanged."""
        return self.run_with_patch_table(PATCH_TABLE)

    def test_fence_passes_when_only_test_support_resolves_the_feature(self):
        code, output = self.run_committed()

        self.assertEqual(code, 0, output)
        self.assertIn(self.FENCE_PASSED, output)

    def test_fence_fails_when_the_release_resolution_has_the_feature(self):
        self.env["CARGO_TREE_RELEASE"] = self.TEST_TREE

        code, output = self.run_committed()

        self.assertEqual(code, 1, output)
        self.assertIn("With no features selected, the workspace resolves", output)
        self.assertNotIn(self.FENCE_PASSED, output)

    def test_fence_control_fails_when_test_support_does_not_resolve_the_feature(self):
        self.env["CARGO_TREE_TEST"] = self.RELEASE_TREE

        code, output = self.run_committed()

        self.assertEqual(code, 1, output)
        self.assertIn("The fence's control failed", output)

    def test_fence_fails_when_the_release_cargo_tree_cannot_resolve(self):
        self.env["CARGO_TREE_RELEASE_EXIT"] = "101"

        code, output = self.run_committed()

        self.assertEqual(code, 1, output)
        self.assertIn(
            "cargo tree could not resolve the workspace's features (output above), "
            "so the fence could not run",
            output,
        )

    def test_fence_fails_when_the_test_support_cargo_tree_cannot_resolve(self):
        # The canned tree still names the feature, so only the exit status
        # can fail this call.
        self.env["CARGO_TREE_TEST_EXIT"] = "101"

        code, output = self.run_committed()

        self.assertEqual(code, 1, output)
        self.assertIn(
            "cargo tree could not resolve the workspace's features with "
            "live-edit/test-support (output above), so the fence could not run",
            output,
        )

    def test_replicas_fence_passes_when_every_feature_leaves_replicas_off(self):
        code, output = self.run_committed()

        self.assertEqual(code, 0, output)
        self.assertIn(self.REPLICAS_PASSED, output)

    def test_replicas_fence_fails_when_every_feature_resolves_replicas(self):
        self.env["CARGO_TREE_ALL"] = "\n".join((self.TEST_TREE, self.REPLICAS_NODE))

        code, output = self.run_committed()

        self.assertEqual(code, 1, output)
        self.assertIn(
            "With every workspace feature selected, the workspace resolves {} with "
            "replicas".format(CRATE),
            output,
        )
        self.assertNotIn(self.REPLICAS_PASSED, output)

    def test_replicas_fence_fails_when_the_all_features_cargo_tree_cannot_resolve(self):
        # The canned tree has no replicas node, so only the exit status can
        # fail this call.
        self.env["CARGO_TREE_ALL_EXIT"] = "101"

        code, output = self.run_committed()

        self.assertEqual(code, 1, output)
        self.assertIn(
            "cargo tree could not resolve the workspace's features with --all-features "
            "(output above), so the fence could not run",
            output,
        )

    #: The feature as the manifest's `test-support` line names it.
    FEATURE_ENTRY = '"{}/{}"'.format(CRATE, "live-edit-retention-report")
    TEST_SUPPORT_LINE = 'test-support = ["dep:libc", {}]'.format(FEATURE_ENTRY)
    DEFAULT_LINE = 'default = ["valkey-tls"]'

    def run_with_manifest_lines(self, test_support_line, default_line):
        """Run against the real `Cargo.toml` with its `test-support` and
        `default` feature lines replaced."""
        for line in (self.TEST_SUPPORT_LINE, self.DEFAULT_LINE):
            self.assertEqual(
                self.real_manifest.count(line + "\n"),
                1,
                "{} no longer carries the line these cases rewrite: {}".format(CARGO_TOML, line),
            )
        manifest = self.real_manifest.replace(
            self.TEST_SUPPORT_LINE + "\n", test_support_line + "\n"
        ).replace(self.DEFAULT_LINE + "\n", default_line + "\n")
        (self.repo / "Cargo.toml").write_text(manifest, encoding="utf-8")
        return self.run_script()

    def manifest_failure(self, hits):
        return "Cargo.toml names {}'s {} feature {} time(s), not exactly once on the test-support".format(
            CRATE, self.RETENTION_FEATURE, hits
        )

    def test_fence_fails_when_the_manifest_does_not_name_the_feature(self):
        code, output = self.run_with_manifest_lines(
            'test-support = ["dep:libc"]', self.DEFAULT_LINE
        )

        self.assertEqual(code, 1, output)
        self.assertIn(self.manifest_failure(0), output)

    def test_fence_fails_when_the_manifests_one_mention_is_off_the_test_support_line(self):
        code, output = self.run_with_manifest_lines(
            'test-support = ["dep:libc"]',
            'default = ["valkey-tls", {}]'.format(self.FEATURE_ENTRY),
        )

        self.assertEqual(code, 1, output)
        self.assertIn(self.manifest_failure(1), output)

    def test_fence_fails_when_the_manifest_names_the_feature_twice(self):
        code, output = self.run_with_manifest_lines(
            self.TEST_SUPPORT_LINE, 'default = ["valkey-tls", {}]'.format(self.FEATURE_ENTRY)
        )

        self.assertEqual(code, 1, output)
        self.assertIn(self.manifest_failure(2), output)


# ── audit-vendored-fred.sh: around cargo audit ────────────────────────────


class AuditVendoredFredTests(_ScriptHarness):
    """The audit script audits the source-less fred as the crates.io release
    it was vendored from, and tells a RustSec finding against fred apart from
    an advisory against cargo-audit itself and from an audit that could not
    run — cargo-audit exits non-zero for all three. On every failure it
    echoes the report, which is the operator's only view of the cause."""

    SCRIPT = AUDIT_SCRIPT

    CHECKSUM = "ab" * 32

    #: The three failure verdicts, one distinctive phrase each.
    ADVISORY = "RustSec reports an advisory (or a yank) against {} {}".format(CRATE, VERSION)
    SELF_ADVISORY = "cargo-audit release itself (report above)"
    UNAUDITED = "no advisory summary was found"

    #: What the self-advisory verdict tells the reader to do, whitespace folded.
    SELF_ADVISORY_FIX = "Upgrade cargo-audit to a release without the advisory"

    #: The line cargo-audit 0.22.2 prints when its own release has an
    #: advisory: `print_self_report` (`cargo-audit/src/presenter.rs`) sends it
    #: through abscissa's `status_err!` under `--deny warnings`, hence the
    #: `error: ` prefix.
    SELF_ADVISORY_LINE = (
        "error: This copy of cargo-audit has known advisories! Upgrade cargo-audit "
        "to the latest version: cargo install --force cargo-audit"
    )

    #: The first line of every cargo-audit report.
    FETCHING = "    Fetching advisory database from `https://github.com/RustSec/advisory-db.git`\n"

    #: A scan of fred that found nothing, before any line a case appends.
    CLEAN_SCAN = (
        FETCHING
        + "      Loaded 900 security advisories (from /tmp/advisory-db)\n"
        + "    Scanning Cargo.lock for vulnerabilities (1 crate dependencies)\n"
    )

    #: A scan of fred with one finding, up to the summary line each case
    #: appends.
    REPORT_HEAD = CLEAN_SCAN + (
        "Crate:     fred\n"
        "Version:   {}\n"
        "ID:        RUSTSEC-2099-0001\n".format(VERSION)
    )

    #: Report lines of runs that never reached the audit, each appended to
    #: `FETCHING`: an advisory-db fetch failure (exit 1) and a lockfile
    #: cargo-audit could not load (exit 2).
    FETCH_FAILURE_LINE = (
        "error: couldn't fetch advisory database: git operation failed: "
        "failed to resolve address for github.com"
    )
    LOCK_LOAD_FAILURE_LINE = "error: couldn't load Cargo.lock: I/O error: permission denied"

    def setUp(self):
        super().setUp()
        self.write_record(VERSION, self.CHECKSUM)
        self.cargo_log = self.tmpdir / "cargo-calls.log"
        self.audited_lock = self.tmpdir / "audited-Cargo.lock"
        self.report = self.tmpdir / "report.txt"
        self.env.update(
            STUB_CARGO_LOG=str(self.cargo_log),
            STUB_CARGO_LOCK_COPY=str(self.audited_lock),
            STUB_CARGO_REPORT=str(self.report),
            STUB_CARGO_VERSION_EXIT="0",
        )
        # `cargo audit --version` exits with STUB_CARGO_VERSION_EXIT; any
        # other call copies the `--file` lockfile out (the script deletes its
        # work dir on exit), prints the report and exits STUB_CARGO_EXIT.
        self.write_stub(
            "cargo",
            'printf \'%s\\n\' "$*" >> "$STUB_CARGO_LOG"\n'
            'if [ "$1" = audit ] && [ "$2" = --version ]; then\n'
            '  exit "$STUB_CARGO_VERSION_EXIT"\n'
            "fi\n"
            "prev=\n"
            'for arg in "$@"; do\n'
            '  if [ "$prev" = --file ]; then cp "$arg" "$STUB_CARGO_LOCK_COPY"; fi\n'
            '  prev=$arg\n'
            "done\n"
            'cat "$STUB_CARGO_REPORT"\n'
            'exit "$STUB_CARGO_EXIT"\n',
        )
        self.write_lock([(CRATE, VERSION, None), ("anyhow", "1.0.0", REGISTRY_SOURCE)])

    def write_lock(self, packages):
        """`Cargo.lock` from `(name, version, source_or_None)` tuples."""
        blocks = ["version = 4\n"]
        for name, version, source in packages:
            block = '[[package]]\nname = "{}"\nversion = "{}"\n'.format(name, version)
            if source is not None:
                block += 'source = "{}"\nchecksum = "{}"\n'.format(source, "0" * 64)
            blocks.append(block)
        (self.repo / "Cargo.lock").write_text("\n".join(blocks), encoding="utf-8")

    def audit(self, report, exit_code, *args):
        """Run the script with `cargo audit` printing `report` and exiting
        `exit_code`."""
        self.report.write_text(report, encoding="utf-8")
        self.env["STUB_CARGO_EXIT"] = str(exit_code)
        return self.run_script(*args)

    def audit_calls(self):
        """The `cargo audit` invocations that audited a lockfile."""
        if not self.cargo_log.exists():
            return []
        return [
            line
            for line in self.cargo_log.read_text(encoding="utf-8").splitlines()
            if " --file " in line
        ]

    def test_lock_without_a_source_less_fred_fails_before_auditing(self):
        self.write_lock([(CRATE, VERSION, REGISTRY_SOURCE)])

        code, output = self.audit("", 0)

        self.assertEqual(code, 1, output)
        self.assertIn("Cargo.lock has no source-less {} package".format(CRATE), output)
        self.assertEqual(self.audit_calls(), [], "nothing was there to audit")

    def test_report_with_an_advisory_summary_fails_as_an_advisory(self):
        for summary in (
            "error: 1 vulnerability found!",
            "error: 2 vulnerabilities found!",
            "error: 1 denied warning found!",
            "error: 3 denied warnings found!",
        ):
            with self.subTest(summary=summary):
                code, output = self.audit(self.REPORT_HEAD + summary + "\n", 1)

                self.assertEqual(code, 1, output)
                self.assertIn(self.ADVISORY, output)
                self.assertNotIn(self.UNAUDITED, output)
                self.assertIn("RUSTSEC-2099-0001", output, "the report must be echoed")

    def test_self_advisory_without_a_fred_summary_fails_as_a_self_advisory(self):
        """The verdict tells the reader to upgrade cargo-audit, in words of its
        own (the echoed report's line says "Upgrade cargo-audit to the latest
        version"), and names no file the published tree lacks, such as the CI
        config that pins the cargo-audit version."""
        code, output = self.audit(self.CLEAN_SCAN + self.SELF_ADVISORY_LINE + "\n", 1)

        self.assertEqual(code, 1, output)
        self.assertIn(self.SELF_ADVISORY, output)
        self.assertNotIn(self.ADVISORY, output)
        self.assertNotIn(self.UNAUDITED, output)
        self.assertIn(self.SELF_ADVISORY_LINE, output, "the report must be echoed")
        self.assertIn(self.SELF_ADVISORY_FIX, " ".join(output.split()))
        self.assertNotIn(".gitlab-ci", output)

    def test_fred_summary_wins_over_a_self_advisory(self):
        """A report carrying both is a finding against fred, whichever line
        comes first."""
        summary = "error: 1 vulnerability found!\n"
        self_line = self.SELF_ADVISORY_LINE + "\n"
        for order, report in (
            ("summary first", self.REPORT_HEAD + summary + self_line),
            ("self-advisory first", self.REPORT_HEAD + self_line + summary),
        ):
            with self.subTest(order=order):
                code, output = self.audit(report, 1)

                self.assertEqual(code, 1, output)
                self.assertIn(self.ADVISORY, output)
                self.assertNotIn(self.SELF_ADVISORY, output)

    def test_report_without_an_advisory_summary_fails_as_unaudited(self):
        for case, exit_code, failure_line in (
            ("advisory-db fetch failure", 1, self.FETCH_FAILURE_LINE),
            ("lockfile load failure", 2, self.LOCK_LOAD_FAILURE_LINE),
        ):
            with self.subTest(case=case):
                code, output = self.audit(self.FETCHING + failure_line + "\n", exit_code)

                self.assertEqual(code, 1, output)
                self.assertIn(self.UNAUDITED, output)
                self.assertNotIn(self.ADVISORY, output)
                self.assertNotIn(self.SELF_ADVISORY, output)
                self.assertIn(failure_line, output, "the report must be echoed")

    def test_clean_audit_passes(self):
        code, output = self.audit("Success No vulnerable packages found\n", 0)

        self.assertEqual(code, 0, output)
        self.assertIn(
            "OK: no RustSec advisory against the vendored {} ({})".format(CRATE, VERSION),
            output,
        )

    def test_audited_lockfile_is_the_crates_io_release_at_the_recorded_checksum(self):
        """What cargo-audit sees is fred alone, registry-sourced, with the
        checksum the record carries: the release the tree was vendored from."""
        code, output = self.audit("", 0)

        self.assertEqual(code, 0, output)
        audited = tomllib.loads(self.audited_lock.read_text(encoding="utf-8"))
        self.assertEqual(
            audited["package"],
            [
                {
                    "name": CRATE,
                    "version": VERSION,
                    "source": REGISTRY_SOURCE,
                    "checksum": self.CHECKSUM,
                }
            ],
        )

    def test_audit_denies_warnings_and_passes_arguments_through(self):
        code, output = self.audit("", 0, "--no-fetch", "--db", "/advisory-db")

        self.assertEqual(code, 0, output)
        calls = self.audit_calls()
        self.assertEqual(len(calls), 1, calls)
        self.assertIn(" --deny warnings ", calls[0])
        self.assertTrue(calls[0].endswith(" --no-fetch --db /advisory-db"), calls[0])
        # `--color never` precedes the caller's arguments, so a `--color` the
        # caller passes still overrides it.
        color_at = calls[0].find(" --color never ")
        self.assertNotEqual(color_at, -1, calls[0])
        self.assertLess(color_at, calls[0].rfind(" --no-fetch --db /advisory-db"), calls[0])

    def test_vendored_version_without_a_record_fails(self):
        self.write_lock([(CRATE, "11.0.0", None)])

        code, output = self.audit("", 0)

        self.assertEqual(code, 1, output)
        self.assertIn(
            "Cargo.lock locks a vendored {} 11.0.0, but vendor/{}-11.0.0.SOURCE.md is "
            "missing.".format(CRATE, CRATE),
            output,
        )
        self.assertEqual(self.audit_calls(), [])

    def test_record_without_a_checksum_fails_before_auditing(self):
        self.write_record(VERSION, "not-a-digest")

        code, output = self.audit("", 0)

        self.assertEqual(code, 1, output)
        self.assertIn(
            "Could not parse the crate checksum from vendor/{}-{}.SOURCE.md".format(
                CRATE, VERSION
            ),
            output,
        )
        self.assertEqual(self.audit_calls(), [])

    def test_missing_cargo_audit_fails_naming_the_install(self):
        self.env["STUB_CARGO_VERSION_EXIT"] = "101"

        code, output = self.audit("", 0)

        self.assertEqual(code, 1, output)
        self.assertIn("cargo-audit is not installed", output)
        self.assertEqual(self.audit_calls(), [])


#: NVIDIA's Apache-2.0 header, verbatim, in the `//` style
#: `tools/license_headers.py` renders for a Rust file.
NVIDIA_RUST_HEADER = """\
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
"""


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VendoredTreeRecordTests(unittest.TestCase):
    """The committed tree, patch and upstream-file manifest agree on which
    files are fred's as published and which are NVIDIA's
    (`vendor/fred-<version>.SOURCE.md` § Upstream source and § License).

    The patch says which files it adds and which it edits: a file whose
    first hunk starts `@@ -0,0` is added, any other is edited. The manifest,
    `vendor/fred-<version>.SHA256SUMS`, lists every file of the published
    crate with its digest. Every listed file the patch does not edit must
    still have that digest in the tree, every file it edits must not, and
    no file it adds may be listed; the added files are the six NVIDIA files,
    and each starts with NVIDIA's Apache-2.0 header. Reads only committed
    files, with no network.
    """

    TREE_DIR = REPO_ROOT / TREE
    MANIFEST = REPO_ROOT / (TREE + ".SHA256SUMS")
    PATCH = REPO_ROOT / (TREE + ".patch")
    #: The files NVIDIA wrote, which the patch adds.
    NVIDIA_FILES = tuple(
        "src/live_edit/{}.rs".format(name)
        for name in ("abandon", "ask", "liveness", "mod", "pool", "retention")
    )
    #: The upstream files fred's patch edits: SOURCE.md § License counts them.
    EDITED_COUNT = 23
    #: Listed files the GitHub source release lacks (`PUBLISH_DROPPED` in
    #: `scripts/check-vendored-fred.sh`); absent, they are skipped, not failed.
    PUBLISH_DROPPED = frozenset({"Cargo.lock", ".vscode/settings.json"})

    def manifest(self):
        """`[(path, digest)]` in file order, read by the license checker's own
        parser (`read_upstream_manifest`), which checks every line as a GNU
        `sha256sum` line and each path as listed once."""
        try:
            return lh.read_upstream_manifest(self.MANIFEST)
        except lh.HeaderError as error:
            self.fail("the upstream-file manifest is malformed: {}".format(error))

    def patch_files(self):
        """`(added, edited)`: the paths the patch names, split by whether
        the file's first hunk starts `@@ -0,0`, read by the license checker's
        own parser (`read_patch_files`), which consumes hunks by their line
        counts, so a removed line that itself starts `-- a/` is never read as
        a file header."""
        try:
            return lh.read_patch_files(self.PATCH)
        except lh.HeaderError as error:
            self.fail("the patch is malformed: {}".format(error))

    def tree_files(self):
        found = set()
        for directory, _, names in os.walk(self.TREE_DIR):
            for name in names:
                found.add(Path(directory, name).relative_to(self.TREE_DIR).as_posix())
        return found

    def test_each_nvidia_file_starts_with_the_verbatim_apache_header(self):
        for relative in self.NVIDIA_FILES:
            with self.subTest(path=relative):
                text = (self.TREE_DIR / relative).read_text(encoding="utf-8")
                # The header, one blank line, then the module's own docs.
                self.assertTrue(
                    text.startswith(NVIDIA_RUST_HEADER + "\n//!"),
                    "{}/{} does not start with NVIDIA's Apache-2.0 header".format(TREE, relative),
                )

    def test_patch_adds_exactly_the_nvidia_files(self):
        added, edited = self.patch_files()

        self.assertEqual(sorted(added), sorted(self.NVIDIA_FILES))
        self.assertEqual(len(edited), self.EDITED_COUNT, edited)

    def test_manifest_is_sorted_unique_relative_paths(self):
        paths = [path for path, _ in self.manifest()]
        encoded = [path.encode("utf-8") for path in paths]

        # Strictly ascending in byte order (`LC_ALL=C sort`): sorted, and no
        # path twice.
        self.assertEqual(encoded, sorted(set(encoded)))
        for path in paths:
            with self.subTest(path=path):
                self.assertFalse(path.startswith(("./", "/")), path)
                self.assertNotIn("..", path.split("/"))

    def test_manifest_lists_no_file_the_patch_adds(self):
        added, _ = self.patch_files()
        listed = {path for path, _ in self.manifest()}

        self.assertEqual(sorted(listed & set(added)), [])

    def test_manifest_lists_every_upstream_file_of_the_tree(self):
        """Every tree file but the added ones is listed, and a listed file
        is absent only where the GitHub source release drops it."""
        added, _ = self.patch_files()
        listed = {path for path, _ in self.manifest()}
        tree = self.tree_files()

        self.assertEqual(sorted(tree - set(added) - listed), [])
        self.assertEqual(sorted(listed - tree - self.PUBLISH_DROPPED), [])

    def test_every_file_the_patch_leaves_alone_has_its_listed_digest(self):
        _, edited = self.patch_files()
        checked = 0
        for path, digest in self.manifest():
            if path in edited or not (self.TREE_DIR / path).is_file():
                continue
            with self.subTest(path=path):
                self.assertEqual(_sha256(self.TREE_DIR / path), digest)
            checked += 1
        # Anti-vacuity: an empty or all-edited manifest would check nothing.
        self.assertGreater(checked, 100)

    def test_every_file_the_patch_edits_differs_from_its_listed_digest(self):
        _, edited = self.patch_files()
        listed = dict(self.manifest())

        self.assertEqual(len(edited), self.EDITED_COUNT)
        for path in edited:
            with self.subTest(path=path):
                self.assertIn(path, listed, "an edited file must be an upstream file")
                self.assertNotEqual(_sha256(self.TREE_DIR / path), listed[path])

    def test_the_license_check_passes_every_file_of_the_tree(self):
        """The license check reads this same record: upstream files pass
        unheaded, and the NVIDIA files pass with their header."""
        scope = lh.Scope(REPO_ROOT)
        self.assertIsNotNone(scope.trees.get(TREE), scope.problems)
        files = [path for path in scope.files if path.startswith(TREE + "/")]
        self.assertGreater(len(files), 100)
        problems = [
            "{}: {}".format(path, problem)
            for path in files
            if (REPO_ROOT / path).is_file()
            for problem in lh.file_problems(scope, path)
        ]
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
