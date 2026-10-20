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

"""Unit tests for `tools/generate_attributions.py`.

Scope: the whole script. It is stdlib-only Python, and after the b.sv5
rewrite its license *text* input is the locally cached `.crate` tarball for
each `(name, version)` pinned in `Cargo.lock`, SHA-256-verified against the
lockfile checksum. These tests drive that entirely through synthetic
tarballs in a `tempfile` cargo home: `CARGO_HOME` is pointed at it and the
memoized `registry_cache_dirs()` is cleared around every test, and
`fetch_url` is replaced by a recording stub — so a test that accidentally
reaches the SPDX tier shows up as a recorded URL rather than as a live
request, and nothing here touches the real registry cache.

What the script is for (issue b.sv5): `supply-chain:attributions-drift`
regenerates `ATTRIBUTIONS.md` and diffs it byte-for-byte against the
committed copy. The previous generator read each crate's license text off
the project's GitHub *default branch* and stamped the wall-clock year into
copyright placeholders, so the job went red with no change in this repo
whenever an upstream landed an unreleased copyright edit (b.tcm), and the
committed artifact carried fabricated years such as
`Copyright 2023-2026 The rustls-webpki authors.` These properties are
load-bearing and each has direct coverage below:

* Text comes from the artifact we redistribute — the *right* one. The
  `(name, version)` pin is not an identity when two registries are
  configured, so every candidate is checksum-verified and a mismatch is a
  hard failure rather than a silent substitution.
* Every top-level license file is emitted, so a dual-licensed crate
  attributes both branches and no branch is elected on the user's behalf.
  The SPDX fallback tier does the same, per declared branch.
* Nothing is derived from the wall clock and nothing is substituted into
  shipped text, so two runs on one input are byte-identical, a year is
  never synthesized, and the verbatim `Copyright [yyyy] [name of copyright
  owner]` appendix lines in the artifact stay verbatim.
* Newline normalization is the only transformation. Fencing is a
  container, not a transformation — license prose full of `#` lines cannot
  be allowed to forge document headings.
* The failure gates fail, and a failed run never overwrites the committed
  document: it lands beside it as `<output>.tmp`.

Three classes reach outside the script and read files off the working
tree, each an anti-vacuity anchor for a property the synthetic fixtures
otherwise assert in a vacuum: `CommittedArtifactTests` (`ATTRIBUTIONS.md`
structure, the absent b.sv5 regression, the *present* Apache
placeholders, a fence-aware section scan, and the per-section
permissiveness gate), `LicenseExpressionTests` (the expression grammar,
plus every expression in `LICENSE-3rdparty.csv`) and `ParseCargoLockTests`
(the real `Cargo.lock` still parses to a checksum-complete package set, and
its source-less packages are all members of the workspace the root
`Cargo.toml` declares or vendored crates).
All assert structure, presence or absence — never counts — so an ordinary
regeneration or dependency bump does not break them.

`ATTRIBUTIONS.md` and `Cargo.lock` are the *committed* anchors: git-tracked,
present in every checkout, readable directly. `LICENSE-3rdparty.csv` is
NOT one of them — it is a gitignored `dd-rust-license-tool write` output,
a generated intermediate the generator consumes on its way to the
document, and it is absent from a fresh checkout. Every read of it
therefore routes through `require_license_csv` below, which skips when the
file is absent and hard-fails instead whenever
`LIVE_EDIT_REQUIRE_LICENSE_CSV` is set to any non-empty value (CI writes
`=1`). That is what splits the CSV-anchored cases across two CI jobs: they
skip in `supply-chain:python-tests` (a `python:3.12-slim` micro-runner
with no Rust toolchain, so it cannot generate the CSV) and are enforced in
`supply-chain:attributions-drift`, which generates the CSV and then
re-runs this module strictly.

Two classes guard that arrangement, because a gate whose default is
"skip" fails silently in exactly the direction nobody watches:
a CI-wiring test (internal: it reads the CI config, so it is not
published) proves CI still runs the strict tier at all, so the
skip tier can never quietly become the only tier, and
`StrictModeSkipGuardTests` proves the strict run has no skips left in it,
so a future case that reaches outside the tree without going through
`require_license_csv` cannot stay green in both jobs.

These tests are stdlib `unittest` (no pytest), matching the sibling
modules and the `supply-chain:python-tests` CI job, which enumerates its
modules explicitly. Run just this module from the repo root with:

    python3 -m unittest tools.tests.test_generate_attributions

or directly:

    python3 tools/tests/test_generate_attributions.py
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib
import io
import os
import re
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


# ── Module loader ─────────────────────────────────────────────────────────


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
TESTS_DIR = Path(__file__).resolve().parent


def _load_generator_module():
    # Make `tools/` importable so `import generate_attributions` works
    # without requiring an installed package. Mirrors the loader in
    # `test_check_bench_thresholds.py`.
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    sys.modules.pop("generate_attributions", None)
    return importlib.import_module("generate_attributions")


# The script has no import-time filesystem side effects: the registry-cache
# glob is resolved on first use and memoized, which is what lets these tests
# point `CARGO_HOME` at a temp tree.
GEN = _load_generator_module()


# ── Shared fixtures ───────────────────────────────────────────────────────


#: The committed anchors, cross-checked by the anchor classes. Both are
#: git-tracked, so both are on disk in every checkout and may be read
#: directly.
ATTRIBUTIONS_PATH = REPO_ROOT / "ATTRIBUTIONS.md"
CARGO_LOCK_PATH = REPO_ROOT / "Cargo.lock"

#: NOT a committed artifact: a gitignored `dd-rust-license-tool write`
#: output (`.gitignore` line 32) that the generator reads as an
#: intermediate. Absent from a fresh checkout, so read it only through
#: `require_license_csv` — opening this path directly raises
#: `FileNotFoundError` anywhere the generator has not been run.
CSV_PATH = REPO_ROOT / "LICENSE-3rdparty.csv"

#: Set to any non-empty value to turn the CSV-anchored tier's skip into a
#: hard failure. `supply-chain:attributions-drift` sets it (to `1`, the
#: repo convention) after generating the CSV. See `strict_csv_required`
#: for why the value itself is not parsed.
REQUIRE_CSV_ENV = "LIVE_EDIT_REQUIRE_LICENSE_CSV"

#: The command that produces `CSV_PATH`, quoted in the skip reason so the
#: reader can act on it without going and finding the CI config.
CSV_GENERATOR_CMD = "dd-rust-license-tool write"

#: The one job that runs the strict half of the CSV-anchored tier.
#: An internal CI-wiring test reads the CI config to prove that job still
#: generates the CSV and still re-runs this module against it.
CI_STRICT_CSV_JOB = "supply-chain:attributions-drift"


def strict_csv_required():
    """Is this run the strict one? True for any non-empty `REQUIRE_CSV_ENV`.

    The single arming predicate, shared by `require_license_csv` and
    `StrictModeSkipGuardTests` so the two can never disagree about which
    tier a given run belongs to.

    Deliberately not a truthy-word parse, and deliberately without a
    `0`-means-off carve-out. This variable exists to turn a skip into a
    failure, so each ambiguous spelling has a safe direction and an unsafe
    one, and they are not symmetric:

    * reading `=true` / `=yes` as "not `1`, so lenient" restores the skip
      in a job that explicitly asked for enforcement. The gate fails
      *open*, every CSV-anchored assertion silently checks nothing, and
      the pipeline stays green while saying nothing about it;
    * reading `=0` as "armed" fails a run an operator meant to relax. That
      is loud, immediate, and names the variable in its own message.

    So: presence arms it. Unsetting the variable is the one documented way
    to allow skips, which is what the failure message says.
    """
    return os.environ.get(REQUIRE_CSV_ENV, "") != ""


def require_license_csv(test):
    """Return `CSV_PATH`, or skip/fail when it is not on disk.

    The single gate every CSV read goes through. Three branches:

    * present — return the path, and the caller asserts as normal;
    * absent while `REQUIRE_CSV_ENV` is set — `fail`, because the caller
      explicitly declared the CSV a precondition. Only the job that
      generates the CSV sets that, so a failure here means the generator
      step did not run or wrote nothing, which would otherwise have
      silently downgraded the strict tier to a skip;
    * absent otherwise — `skipTest`, because a gitignored generated
      intermediate being missing is the *expected* state of a fresh
      checkout, not a defect in the code under test.

    The skip is the load-bearing default and the reason it must stay
    loud: a bare `FileNotFoundError` here reads as a broken test, and the
    remedy — run the generator, or accept the skip because CI enforces it
    elsewhere — is not guessable from the traceback.
    """
    if CSV_PATH.is_file():
        return CSV_PATH

    if strict_csv_required():
        test.fail(
            "{} is absent, but {}={!r} declares it a precondition (any "
            "non-empty value arms strict mode; unset the variable to allow "
            "skips). `{}` runs `{}` before this strict run, so an absent CSV "
            "here means that generator step did not run or produced no file — "
            "the anti-vacuity and permissiveness gates would have been skipped "
            "in the one job that is supposed to enforce them.".format(
                CSV_PATH,
                REQUIRE_CSV_ENV,
                os.environ.get(REQUIRE_CSV_ENV, ""),
                CI_STRICT_CSV_JOB,
                CSV_GENERATOR_CMD,
            )
        )

    test.skipTest(
        "{} is absent. It is not a committed artifact: it is a gitignored "
        "`{}` output, generated on demand. Run `{}` from the repo root to "
        "exercise this check locally. CI does not rely on this run — `{}` "
        "generates the CSV and re-runs this module with {}=1, where the "
        "skip becomes a hard failure.".format(
            CSV_PATH,
            CSV_GENERATOR_CMD,
            CSV_GENERATOR_CMD,
            CI_STRICT_CSV_JOB,
            REQUIRE_CSV_ENV,
        )
    )


#: The exact fabricated-copyright line the pre-b.sv5 generator wrote into
#: `ATTRIBUTIONS.md`: an upstream year range with the wall-clock year
#: substituted into its tail.
FABRICATED_COPYRIGHT_LINE = "Copyright 2023-2026 The rustls-webpki authors."

#: A crate registry `source` value, as `Cargo.lock` writes it. Only
#: `registry+`-sourced packages have a `.crate` artifact and a checksum.
REGISTRY_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"

#: CSV row for the two-version `multi` crate built by
#: `_GeneratorHarness.arrange_two_versions`.
MULTI_CSV_ROW = ("multi", "https://example.invalid/multi", "MIT", "M Author")

#: Stand-in checksum for a locked version with no tarball on disk. The
#: lockfile always records one, so omitting it would trip the integrity
#: gate instead of the missing-artifact gate under test.
PLACEHOLDER_CHECKSUM = "0" * 64

#: A syntactically valid checksum that matches no fixture's bytes.
WRONG_CHECKSUM = "f" * 64

#: `write_lock(checksums={...: OMIT})` drops the `checksum` line entirely.
OMIT = object()

#: The heading of the failure bucket for a source-less `Cargo.lock` package
#: nothing attributes. The bucket's own line is what tells this failure apart
#: from the others a run can report.
UNATTRIBUTED_PATH_PACKAGES_ERROR = (
    "ERROR: source-less Cargo.lock packages that are neither workspace "
    "members nor VENDORED_CRATES at their locked version:"
)

MIT_TEXT = (
    "MIT License\n"
    "\n"
    "Copyright (c) 2015 Some Upstream Author\n"
    "\n"
    "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
    "of this software and associated documentation files (the \"Software\"), to deal\n"
    "in the Software without restriction.\n"
)

APACHE_TEXT = (
    "                                 Apache License\n"
    "                           Version 2.0, January 2004\n"
    "                        http://www.apache.org/licenses/LICENSE-2.0\n"
    "\n"
    "Licensed under the Apache License, Version 2.0 (the \"License\").\n"
)

#: A shipped `LICENSE-APACHE` including the boilerplate appendix. Its two
#: placeholders are the ones `fill_copyright_placeholders` would rewrite if
#: it ever ran on tarball text — `[name of copyright owner]` is in
#: `COPYRIGHT_HOLDER_PATTERNS`, and `[yyyy]` was a year-substitution target
#: before b.sv5. Both appear verbatim ~99 times in the committed artifact.
APACHE_APPENDIX_TEXT = (
    APACHE_TEXT
    + "\n"
    + "   APPENDIX: How to apply the Apache License to your work.\n"
    + "\n"
    + "   Copyright [yyyy] [name of copyright owner]\n"
    + "\n"
    + "   Licensed under the Apache License, Version 2.0 (the \"License\");\n"
)

#: The crate `EMBEDDED_LICENSE_FILES` declares a nested notice for, the path
#: it declares, and that notice's text (jemalloc's BSD-2-Clause `COPYING`,
#: abridged). The fixtures use the real crate name so they exercise the real
#: declaration, not a patched copy of it.
EMBEDDING_CRATE = "tikv-jemalloc-sys"
EMBEDDED_PATH = "jemalloc/COPYING"
EMBEDDED_EXPRESSION = "(MIT OR Apache-2.0) AND BSD-2-Clause"
JEMALLOC_COPYING_TEXT = (
    "Unless otherwise specified, files in the jemalloc source distribution are\n"
    "subject to the following license:\n"
    "Copyright (C) 2002-present Jason Evans <jasone@canonware.com>.\n"
    "All rights reserved.\n"
    "\n"
    "Redistribution and use in source and binary forms, with or without\n"
    "modification, are permitted provided that the following conditions are met:\n"
)

#: Opens with the two-line heading a real GNU license text opens with,
#: which is what the narrowed copyleft signatures require.
GPL_TEXT = (
    "                    GNU GENERAL PUBLIC LICENSE\n"
    "                       Version 3, 29 June 2007\n"
    "\n"
    " Copyright (C) 2007 Free Software Foundation, Inc.\n"
)

#: SPDX canonical texts as the pinned `license-list-data` tag ships them:
#: a `<year>` placeholder alongside the holder placeholder. The whole point
#: of the fallback tier is that the year survives.
SPDX_MIT_TEMPLATE = (
    "MIT License\n"
    "\n"
    "Copyright (c) <year> <copyright holders>\n"
    "\n"
    "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
)

SPDX_APACHE_TEMPLATE = (
    "                                 Apache License\n"
    "                           Version 2.0, January 2004\n"
    "\n"
    "   Copyright [yyyy] [name of copyright owner]\n"
)


def _reverse_spdx_urls():
    """`{url: license_id}` for the script's pinned SPDX URL table."""
    return {url: license_id for license_id, url in GEN.SPDX_URLS.items()}


# ── Fence-aware document scanning ─────────────────────────────────────────
#
# License text is emitted inside backtick fences precisely so that a `#`
# line in a shipped license cannot forge a document heading. A test that
# scanned the document with a plain `startswith` could not tell the
# difference — which is the property under test — so every structural
# assertion goes through these two helpers.


HEADING_PREFIXES = ("## ", "### ")
FENCE_RE = re.compile(r"`{3,}$")

#: The character class an SPDX short identifier is drawn from. Used only
#: to tell "this line is an expression" from "this line is prose, a URL or
#: a fence" — `split_license_expr` handles the operators and validates
#: nothing, so this is what stops an extraction bug from being reported as
#: a licensing problem.
SPDX_ID_RE = re.compile(r"[A-Za-z0-9.+-]+")


def looks_like_spdx_expression(expression):
    """True when every operand of `expression` is shaped like an SPDX id."""
    ids = GEN.split_license_expr(expression)
    return bool(ids) and all(SPDX_ID_RE.fullmatch(lic) for lic in ids)


def scan_document(text):
    """Fence-aware structural scan of an emitted document.

    Returns `(headings, fenced_heading_lines, unclosed_fence)`: the
    document-level heading lines, a count of heading-*looking* lines that
    are safely inside a fenced block, and the fence left open at EOF
    (`None` when every fence closed).
    """
    headings = []
    fenced_headings = 0
    fence = None
    for line in text.split("\n"):
        if fence is None:
            if FENCE_RE.fullmatch(line):
                fence = line
            elif line.startswith(HEADING_PREFIXES):
                headings.append(line)
        elif line == fence:
            fence = None
        elif line.startswith(HEADING_PREFIXES):
            fenced_headings += 1
    return headings, fenced_headings, fence


def section_bodies(text):
    """`{component: body}` for every document-level `## ` section."""
    bodies = {}
    current = None
    buf = []
    fence = None
    for line in text.split("\n"):
        if fence is None and line.startswith("## "):
            if current is not None:
                bodies[current] = "\n".join(buf)
            current, buf = line[len("## "):], []
            continue
        if fence is None:
            if FENCE_RE.fullmatch(line):
                fence = line
        elif line == fence:
            fence = None
        if current is not None:
            buf.append(line)
    if current is not None:
        bodies[current] = "\n".join(buf)
    return bodies


class _GeneratorHarness(unittest.TestCase):
    """Per-test temp cargo home, CSV/lock inputs and network stub.

    The script keeps process-global state — the memoized registry-cache
    glob, the SPDX text cache and its per-id fetch locks, the rate-limit
    backoff. All of it is reset here so no test can depend on another
    having run.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)

        # Registered first so it runs *last*: the memo must be dropped
        # after `CARGO_HOME` is restored, or the next test inherits a
        # tuple of paths into this test's deleted temp tree.
        self.addCleanup(GEN.registry_cache_dirs.cache_clear)

        self.cargo_home = self.tmpdir / "cargo-home"
        self.cache_dir = self.add_cache_dir("index.crates.io-6f17d22bba15001f")
        env_patcher = mock.patch.dict(os.environ, {"CARGO_HOME": str(self.cargo_home)})
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        GEN.registry_cache_dirs.cache_clear()

        #: `(name, version)` → SHA-256 of the tarball `write_crate` wrote,
        #: so `write_lock` can record the checksum the script verifies.
        self.crate_checksums = {}

        # SPDX tier: `spdx_texts` is the canned corpus, `fetched` records
        # every URL the script asked for. Tests that must not reach the
        # network assert `fetched == []`.
        self.spdx_texts = {}
        self.fetched = []
        fetch_patcher = mock.patch.object(GEN, "fetch_url", self._fake_fetch_url)
        fetch_patcher.start()
        self.addCleanup(fetch_patcher.stop)

        self.addCleanup(self.reset_spdx_probe)
        self.reset_spdx_probe()
        GEN.reset_rate_limit()

    def _fake_fetch_url(self, url, retries=3):
        self.fetched.append(url)
        license_id = _reverse_spdx_urls().get(url)
        return self.spdx_texts.get(license_id)

    def reset_spdx_probe(self):
        """Clear the fetch log and the script's SPDX caches.

        Called between `subTest` iterations so a `fetched` assertion is
        about that iteration alone — `license_cache` would otherwise serve
        iteration 2 from iteration 1's fetch and the log would not grow.
        """
        del self.fetched[:]
        GEN.license_cache.clear()
        GEN.license_fetch_locks.clear()

    # ── Input builders ────────────────────────────────────────────────

    def add_cache_dir(self, name):
        """Create another registry cache dir under the temp `CARGO_HOME`.

        Two configured registries is the normal case this container is in
        (crates.io plus an internal mirror), and it is what makes the
        checksum verification load-bearing. Drops the glob memo so the new
        directory is picked up.
        """
        path = self.cargo_home / "registry" / "cache" / name
        path.mkdir(parents=True, exist_ok=True)
        GEN.registry_cache_dirs.cache_clear()
        return path

    def write_crate(self, name, version, files=(), extra_members=(), cache_dir=None, record=True):
        """Write a synthetic `<name>-<version>.crate` into a cache dir.

        `files` maps a member path *relative to the crate root* to its
        content (`str` is encoded UTF-8; pass `bytes` to control the exact
        line endings). `extra_members` names empty non-license members —
        `Cargo.toml`, `README.md`, `src/lib.rs` — that the enumeration must
        ignore. With `record` the tarball's SHA-256 is remembered so
        `write_lock` emits the matching checksum; pass `record=False` for a
        decoy artifact the lockfile must reject. Returns the path.
        """
        target_dir = self.cache_dir if cache_dir is None else cache_dir
        archive = target_dir / "{}-{}.crate".format(name, version)
        root = "{}-{}".format(name, version)
        members = dict(files)
        for extra in extra_members:
            members.setdefault(extra, "")
        with tarfile.open(archive, "w:gz") as tf:
            for relpath, content in members.items():
                payload = content.encode("utf-8") if isinstance(content, str) else content
                info = tarfile.TarInfo("{}/{}".format(root, relpath))
                info.size = len(payload)
                info.mtime = 0
                tf.addfile(info, io.BytesIO(payload))
        if record:
            self.crate_checksums[(name, version)] = self.sha256_of(archive)
        return archive

    def write_raw_archive(self, name, version, data, cache_dir=None, record=True):
        """Put arbitrary bytes where a `.crate` belongs (corruption tests)."""
        target_dir = self.cache_dir if cache_dir is None else cache_dir
        archive = target_dir / "{}-{}.crate".format(name, version)
        archive.write_bytes(data)
        if record:
            self.crate_checksums[(name, version)] = self.sha256_of(archive)
        return archive

    @staticmethod
    def sha256_of(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def checksum_of(self, name, version):
        """The recorded checksum, as `Cargo.lock` would carry it."""
        return self.crate_checksums.get((name, version), PLACEHOLDER_CHECKSUM)

    def write_csv(self, rows, name="LICENSE-3rdparty.csv"):
        """`rows` are `(component, origin, license, copyright)` tuples."""
        path = self.tmpdir / name
        lines = ["Component,Origin,License,Copyright"]
        for component, origin, license_expr, copyright_info in rows:
            lines.append(
                ",".join(
                    ['"{}"'.format(field.replace('"', '""')) for field in
                     (component, origin, license_expr, copyright_info)]
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    def write_lock(self, packages, checksums=None, name="Cargo.lock"):
        """`packages` are `(name, version, source_or_None)` tuples.

        Registry packages get a `checksum` line: the recorded SHA-256 of
        the tarball `write_crate` wrote, or `PLACEHOLDER_CHECKSUM` when no
        artifact was written. `checksums` overrides per `(name, version)`;
        the `OMIT` sentinel drops the line, modelling a malformed lockfile.
        """
        overrides = checksums or {}
        blocks = ["# This file is automatically @generated by Cargo.\nversion = 4\n"]
        for pkg_name, version, source in packages:
            block = [
                "[[package]]",
                'name = "{}"'.format(pkg_name),
                'version = "{}"'.format(version),
            ]
            if source is not None:
                block.append('source = "{}"'.format(source))
                if source.startswith("registry+"):
                    checksum = overrides.get(
                        (pkg_name, version), self.checksum_of(pkg_name, version)
                    )
                    if checksum is not OMIT:
                        block.append('checksum = "{}"'.format(checksum))
            blocks.append("\n".join(block) + "\n")
        return self.write_lock_text("\n".join(blocks), name=name)

    def write_lock_text(self, text, name="Cargo.lock"):
        path = self.tmpdir / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def arrange_two_versions(self, text_v1=MIT_TEXT, text_v2=MIT_TEXT, cached=("1.0.0", "2.0.0")):
        """Lock `multi` at 1.0.0 and 2.0.0, caching the `cached` subset.

        Returns `(csv_path, lock_path)`, ready to splat into
        `run_generator`. Both versions are always *locked*; dropping one
        from `cached` models a registry cache `cargo fetch --locked` has
        not filled.
        """
        texts = {"1.0.0": text_v1, "2.0.0": text_v2}
        for version in cached:
            self.write_crate("multi", version, files={"LICENSE": texts[version]})
        csv_path = self.write_csv([MULTI_CSV_ROW])
        lock_path = self.write_lock(
            [("multi", version, REGISTRY_SOURCE) for version in ("1.0.0", "2.0.0")]
        )
        return csv_path, lock_path

    def locked_map(self, name, *versions):
        """`{version: checksum}` as `parse_cargo_lock` would return it."""
        return {version: self.checksum_of(name, version) for version in versions}

    # ── Driver ────────────────────────────────────────────────────────

    def run_generator(self, csv_path, lock_path, output_name="ATTRIBUTIONS.md"):
        """Invoke `main()` with a synthesised argv.

        Returns `(exit_code, document, stderr)`. argv[0] is the script name
        the real CLI passes through, so `main`'s slicing is exercised as
        shipped. A clean return is reported as exit code 0.

        `document` is the text the run produced: the target on success, the
        `.tmp` sibling on failure (a failed run leaves the target
        untouched). `self.target_path` / `self.tmp_path` expose both paths
        for the tests asserting on the atomic write itself.
        """
        self.target_path = self.tmpdir / output_name
        self.tmp_path = self.tmpdir / (output_name + ".tmp")
        argv = ["generate_attributions.py", csv_path, str(self.target_path), lock_path]
        stdout, stderr = io.StringIO(), io.StringIO()
        code = 0
        with mock.patch.object(sys, "argv", argv):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    GEN.main()
                except SystemExit as exit_exc:
                    code = exit_exc.code if exit_exc.code is not None else 0
        source = self.tmp_path if code != 0 else self.target_path
        # `read_bytes`, not `read_text`: text mode applies universal-newline
        # translation, which would silently rewrite a CRLF the generator
        # emitted into LF and make the normalization assertions vacuous.
        document = source.read_bytes().decode("utf-8") if source.exists() else ""
        return code, document, stderr.getvalue()

    # ── Assertion helpers ─────────────────────────────────────────────

    def section(self, document, component):
        """The `## <component>` section body, fence-aware."""
        bodies = section_bodies(document)
        self.assertIn(
            component, bodies, "no section for {} in the document".format(component)
        )
        return bodies[component]

    def sub_headings(self, document, component):
        """Document-level `### ` labels inside one component's section."""
        headings, _, _ = scan_document(self.section(document, component))
        return [line[len("### "):] for line in headings if line.startswith("### ")]


# ── Tarball license-file enumeration ──────────────────────────────────────


class ReadCrateLicenseFilesTests(_GeneratorHarness):
    """`read_crate_license_files` — which tarball members count as license
    text. The tarball's members are enumerated rather than probed for a
    fixed filename list, so crate-specific names are picked up without a
    per-crate table."""

    def read(self, name, version):
        """Read a crate against the checksum recorded for its bytes."""
        return GEN.read_crate_license_files(name, version, self.checksum_of(name, version))

    def test_each_recognized_family_is_collected_case_insensitively(self):
        self.write_crate(
            "alpha",
            "1.0.0",
            files={
                "LICENSE": "plain",
                "license-apache-2.0": "lowercased",
                "COPYING": "copying",
                "UNLICENSE": "unlicense",
                "COPYRIGHT": "copyright",
                "LICENCE.BoringSSL": "british spelling, crate-specific suffix",
            },
        )

        files, integrity_error = self.read("alpha", "1.0.0")

        self.assertIsNone(integrity_error)
        self.assertEqual(
            sorted(files),
            ["COPYING", "COPYRIGHT", "LICENCE.BoringSSL", "LICENSE", "UNLICENSE",
             "license-apache-2.0"],
        )

    def test_non_license_top_level_members_are_ignored(self):
        self.write_crate(
            "alpha",
            "1.0.0",
            files={"LICENSE": MIT_TEXT},
            extra_members=["Cargo.toml", "README.md", "CHANGELOG.md", "build.rs"],
        )

        files, _ = self.read("alpha", "1.0.0")

        self.assertEqual(sorted(files), ["LICENSE"])

    def test_nested_license_file_is_not_collected(self):
        """Only the crate root is attribution-bearing. A `LICENSE` under
        `src/` or a vendored subtree belongs to code the crate embeds, not
        to the crate we redistribute."""
        self.write_crate(
            "alpha",
            "1.0.0",
            files={"src/LICENSE": MIT_TEXT, "vendor/dep/LICENSE-MIT": MIT_TEXT},
        )

        files, _ = self.read("alpha", "1.0.0")

        self.assertEqual(files, {})

    def test_absent_archive_is_distinguished_from_a_licenseless_one(self):
        """`None` files with no integrity error means "run `cargo fetch
        --locked`" (a hard failure); `{}` means "this crate genuinely ships
        no license file" (the SPDX fallback). Collapsing the two would turn
        a cold cache into silent SPDX boilerplate."""
        self.write_crate("present", "1.0.0", extra_members=["Cargo.toml"])

        absent = self.read("absent", "1.0.0")
        present = self.read("present", "1.0.0")

        self.assertEqual(absent, (None, None))
        self.assertEqual(present, ({}, None))

    def test_line_endings_are_normalized_to_lf(self):
        """The one transformation applied to shipped text: git checks the
        committed `ATTRIBUTIONS.md` out as LF, so a CRLF-shipping crate
        would otherwise make the drift job diff LF against CRLF forever.
        Lone CR is normalized too — `normalize_newlines` handles both."""
        cases = [
            ("crlf", b"MIT License\r\n\r\nCopyright (c) 2015 Someone\r\n"),
            ("lone_cr", b"MIT License\r\rCopyright (c) 2015 Someone\r"),
            ("mixed", b"MIT License\r\n\rCopyright (c) 2015 Someone\n"),
        ]
        for label, raw in cases:
            with self.subTest(line_endings=label):
                name = "crlf-{}".format(label)
                self.write_crate(name, "1.0.0", files={"LICENSE": raw})

                files, _ = self.read(name, "1.0.0")

                self.assertNotIn("\r", files["LICENSE"])
                self.assertEqual(
                    files["LICENSE"], "MIT License\n\nCopyright (c) 2015 Someone\n"
                )

    def test_non_utf8_bytes_are_replaced_rather_than_raising(self):
        """A few crates ship Latin-1 copyright lines. `errors="replace"`
        keeps one bad byte from aborting the whole run; the substitution
        character is what lands in the artifact."""
        self.write_crate("latin1", "1.0.0", files={"LICENSE": b"Copyright \xa9 2015 Someone\n"})

        files, _ = self.read("latin1", "1.0.0")

        # U+FFFD written as an escape: a literal replacement character trips
        # SonarQube's "invalid character for encoding UTF-8" check.
        self.assertEqual(files["LICENSE"], "Copyright \ufffd 2015 Someone\n")

    def test_oversized_license_member_is_an_integrity_error(self):
        """A license member above the byte cap is refused rather than
        decompressed into memory; the operator-facing error names the member
        and the cap so the artifact can be inspected."""
        too_big = b"x" * (1024 * 1024 + 1)
        self.write_crate("bloated", "1.0.0", files={"LICENSE": too_big})

        files, integrity_error = self.read("bloated", "1.0.0")

        self.assertIsNone(files)
        self.assertIn("bloated-1.0.0/LICENSE", integrity_error)
        self.assertIn("license-file cap", integrity_error)

    def test_license_member_at_the_cap_is_still_read(self):
        """The cap is exclusive: a member exactly at the limit is read."""
        at_cap = b"y" * (1024 * 1024)
        self.write_crate("wide", "1.0.0", files={"LICENSE": at_cap})

        files, integrity_error = self.read("wide", "1.0.0")

        self.assertIsNone(integrity_error)
        self.assertEqual(len(files["LICENSE"]), 1024 * 1024)


# ── Declared embedded license files ───────────────────────────────────────


class EmbeddedLicenseFileTests(_GeneratorHarness):
    """`EMBEDDED_LICENSE_FILES`: a crate that compiles an embedded library
    into our binary owes that library's notice, which lives below the crate
    root. tikv-jemalloc-sys builds jemalloc (BSD-2-Clause) into the service
    binary as its global allocator, and its root ships only the binding's
    MIT / Apache-2.0 texts."""

    def write_embedding_crate(self, with_embedded=True, extra_files=None):
        files = {"LICENSE-MIT": MIT_TEXT, "LICENSE-APACHE": APACHE_TEXT}
        if with_embedded:
            files[EMBEDDED_PATH] = JEMALLOC_COPYING_TEXT
        files.update(extra_files or {})
        self.write_crate(EMBEDDING_CRATE, "0.6.1", files=files)

    def arrange_run(self, with_embedded=True):
        self.write_embedding_crate(with_embedded=with_embedded)
        csv_path = self.write_csv(
            [(EMBEDDING_CRATE, "https://example.invalid/jemalloc", EMBEDDED_EXPRESSION, "J Author")]
        )
        lock_path = self.write_lock([(EMBEDDING_CRATE, "0.6.1", REGISTRY_SOURCE)])
        return csv_path, lock_path

    def test_declared_nested_file_is_read_under_its_path(self):
        self.write_embedding_crate()

        files, integrity_error = GEN.read_crate_license_files(
            EMBEDDING_CRATE, "0.6.1", self.checksum_of(EMBEDDING_CRATE, "0.6.1")
        )

        self.assertIsNone(integrity_error)
        self.assertEqual(sorted(files), ["LICENSE-APACHE", "LICENSE-MIT", EMBEDDED_PATH])
        self.assertEqual(files[EMBEDDED_PATH], JEMALLOC_COPYING_TEXT)

    def test_declared_nested_file_is_emitted_under_its_path_heading(self):
        code, document, stderr = self.run_generator(*self.arrange_run())

        self.assertEqual(code, 0, stderr)
        self.assertEqual(
            self.sub_headings(document, EMBEDDING_CRATE),
            ["LICENSE-APACHE", "LICENSE-MIT", EMBEDDED_PATH],
        )
        section = self.section(document, EMBEDDING_CRATE)
        self.assertIn("Jason Evans", section)
        self.assertIn("\n{}\n".format(EMBEDDED_EXPRESSION), section)

    def test_missing_declared_file_fails_and_leaves_the_committed_document(self):
        """A crate-layout change that moves the notice must fail the run,
        not silently drop the notice from the shipped attributions."""
        csv_path, lock_path = self.arrange_run(with_embedded=False)
        committed = "# Attributions\n\n## previously-committed\n\nMIT\n"
        target = self.tmpdir / "ATTRIBUTIONS.md"
        target.write_text(committed, encoding="utf-8")

        code, _, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 1)
        self.assertIn("failed verification", stderr)
        self.assertIn(EMBEDDED_PATH, stderr)
        self.assertIn("EMBEDDED_LICENSE_FILES", stderr)
        self.assertEqual(target.read_bytes().decode("utf-8"), committed)

    def test_undeclared_crate_at_the_declared_path_is_not_collected(self):
        """The declaration is keyed by crate: the same nested path in any
        other crate is vendored code we do not build in."""
        self.write_crate("alpha", "1.0.0", files={"LICENSE": MIT_TEXT, EMBEDDED_PATH: GPL_TEXT})

        files, integrity_error = GEN.read_crate_license_files(
            "alpha", "1.0.0", self.checksum_of("alpha", "1.0.0")
        )

        self.assertIsNone(integrity_error)
        self.assertEqual(sorted(files), ["LICENSE"])

    def test_declared_crates_other_nested_license_files_are_not_collected(self):
        """Only the declared path is added; the rest of the embedded tree is
        still not attribution-bearing."""
        self.write_embedding_crate(
            extra_files={"jemalloc/LICENSE": GPL_TEXT, "jemalloc/test/COPYING": GPL_TEXT}
        )

        files, _ = GEN.read_crate_license_files(
            EMBEDDING_CRATE, "0.6.1", self.checksum_of(EMBEDDING_CRATE, "0.6.1")
        )

        self.assertEqual(sorted(files), ["LICENSE-APACHE", "LICENSE-MIT", EMBEDDED_PATH])


# ── Registry-cache resolution ─────────────────────────────────────────────


class RegistryCacheResolutionTests(_GeneratorHarness):
    """`registry_cache_dirs` + `find_crate_archive` — the part that reads
    the real environment.

    Without these the `$CARGO_HOME` glob CI depends on would have no
    coverage: a typo in it surfaces as every crate missing from the cache,
    in CI, on a job that takes minutes to reach that point.
    """

    def test_cache_dirs_are_globbed_from_cargo_home(self):
        mirror = self.add_cache_dir("some-mirror-0123456789abcdef")
        # A sibling of `cache/` that the glob must not pick up.
        (self.cargo_home / "registry" / "src").mkdir(parents=True, exist_ok=True)

        self.assertEqual(
            GEN.registry_cache_dirs(), tuple(sorted([str(self.cache_dir), str(mirror)]))
        )

    def test_cache_dirs_are_memoized_until_explicitly_cleared(self):
        """The memo is why `CARGO_HOME` can be resolved lazily at all; it
        is also why every test here clears it. Pinned so a change back to
        an eagerly-resolved constant is a deliberate one."""
        first = GEN.registry_cache_dirs()
        late = self.cargo_home / "registry" / "cache" / "added-after-the-first-call"
        late.mkdir(parents=True)

        self.assertEqual(GEN.registry_cache_dirs(), first)

        GEN.registry_cache_dirs.cache_clear()

        self.assertIn(str(late), GEN.registry_cache_dirs())

    def test_cache_dirs_fall_back_to_the_home_cargo_directory(self):
        """Local runs rarely set `CARGO_HOME`; CI always does."""
        home = self.tmpdir / "home"
        cache = home / ".cargo" / "registry" / "cache" / "index.crates.io-6f17d22bba15001f"
        cache.mkdir(parents=True)

        with mock.patch.dict(os.environ, {"HOME": str(home)}):
            os.environ.pop("CARGO_HOME", None)
            GEN.registry_cache_dirs.cache_clear()

            self.assertEqual(GEN.registry_cache_dirs(), (str(cache),))

    def test_archive_is_found_in_a_later_cache_dir(self):
        mirror = self.add_cache_dir("z-mirror-fedcba9876543210")
        self.write_crate("beta", "1.0.0", files={"LICENSE": MIT_TEXT}, cache_dir=mirror)

        path, integrity_error = GEN.find_crate_archive(
            "beta", "1.0.0", self.checksum_of("beta", "1.0.0")
        )

        self.assertIsNone(integrity_error)
        self.assertEqual(path, str(mirror / "beta-1.0.0.crate"))

    def test_archive_lookup_is_keyed_by_version_not_just_name(self):
        self.write_crate("beta", "2.0.0", files={"LICENSE": MIT_TEXT})

        self.assertEqual(
            GEN.find_crate_archive("beta", "1.0.0", PLACEHOLDER_CHECKSUM), (None, None)
        )

    def test_archive_absent_from_every_cache_dir_is_not_an_integrity_error(self):
        """"Not cached" is not "substituted": the operator action is
        `cargo fetch --locked`, and the two are reported separately."""
        self.add_cache_dir("z-mirror-fedcba9876543210")

        self.assertEqual(
            GEN.find_crate_archive("beta", "1.0.0", PLACEHOLDER_CHECKSUM), (None, None)
        )


# ── Checksum integrity ────────────────────────────────────────────────────


class ChecksumIntegrityTests(_GeneratorHarness):
    """The `(name, version)` pin is not an identity when more than one
    registry is configured, so the chosen artifact is SHA-256-verified
    against `Cargo.lock` before its bytes are read."""

    CSV_ROW = ("alpha", "https://example.invalid/alpha", "MIT", "A Author")

    def arrange(self, checksums=None):
        csv_path = self.write_csv([self.CSV_ROW])
        lock_path = self.write_lock([("alpha", "1.0.0", REGISTRY_SOURCE)], checksums=checksums)
        return csv_path, lock_path

    def test_verified_artifact_is_read(self):
        """Anti-vacuity for the rejections below: the ordinary path must
        not be failing verification for some unrelated reason."""
        self.write_crate("alpha", "1.0.0", files={"LICENSE": MIT_TEXT})

        code, document, _ = self.run_generator(*self.arrange())

        self.assertEqual(code, 0)
        self.assertIn("Permission is hereby granted", self.section(document, "alpha"))

    def test_checksum_mismatch_fails_in_its_own_bucket(self):
        """Distinct from "missing from the cache": the artifact is present,
        it is simply not the one the lockfile pins."""
        self.write_crate("alpha", "1.0.0", files={"LICENSE": MIT_TEXT})

        code, _, stderr = self.run_generator(
            *self.arrange(checksums={("alpha", "1.0.0"): WRONG_CHECKSUM})
        )

        self.assertEqual(code, 1)
        self.assertIn("cached crate artifacts failed verification", stderr)
        self.assertIn("does not match the Cargo.lock checksum", stderr)
        self.assertNotIn("missing from the registry cache", stderr)

    def test_correct_artifact_in_a_later_cache_dir_wins_over_a_decoy(self):
        """Every candidate is checked, not just the first by filename: a
        name collision in the first registry degrades to "keep looking"
        rather than attributing text from an artifact we do not ship."""
        decoy_dir = self.add_cache_dir("a-decoy-registry-0000000000000000")
        real_dir = self.add_cache_dir("z-real-registry-1111111111111111")
        self.write_crate(
            "alpha",
            "1.0.0",
            files={"LICENSE": "DECOY TEXT, NOT WHAT WE SHIP\n"},
            cache_dir=decoy_dir,
            record=False,
        )
        self.write_crate("alpha", "1.0.0", files={"LICENSE": MIT_TEXT}, cache_dir=real_dir)

        code, document, _ = self.run_generator(*self.arrange())

        self.assertEqual(code, 0)
        section = self.section(document, "alpha")
        self.assertIn("Permission is hereby granted", section)
        self.assertNotIn("DECOY TEXT", section)

    def test_lockfile_without_a_checksum_is_a_hard_failure(self):
        """A registry package always records one, so its absence means the
        lockfile is not something we can verify against — not a licence to
        read the artifact unverified."""
        self.write_crate("alpha", "1.0.0", files={"LICENSE": MIT_TEXT})

        code, _, stderr = self.run_generator(
            *self.arrange(checksums={("alpha", "1.0.0"): OMIT})
        )

        self.assertEqual(code, 1)
        self.assertIn("cached crate artifacts failed verification", stderr)
        self.assertIn("no `checksum` recorded in Cargo.lock", stderr)

    def test_unverifiable_crate_does_not_take_the_spdx_fallback(self):
        """Boilerplate here would paper over an artifact we could not
        verify — the same reasoning as the unlocked-component bucket."""
        self.spdx_texts["MIT"] = SPDX_MIT_TEMPLATE
        self.write_crate("alpha", "1.0.0", files={"LICENSE": MIT_TEXT})

        code, _, _ = self.run_generator(
            *self.arrange(checksums={("alpha", "1.0.0"): WRONG_CHECKSUM})
        )

        self.assertEqual(code, 1)
        self.assertEqual(self.fetched, [])


# ── Section assembly ──────────────────────────────────────────────────────


class LicenseBlockEmissionTests(_GeneratorHarness):
    """End-to-end shape of an emitted `## <component>` section."""

    def test_single_license_file_is_emitted_without_a_sub_heading(self):
        self.write_crate("alpha", "1.0.0", files={"LICENSE": MIT_TEXT})
        csv_path = self.write_csv([("alpha", "https://example.invalid/alpha", "MIT", "A Author")])
        lock_path = self.write_lock([("alpha", "1.0.0", REGISTRY_SOURCE)])

        code, document, _ = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 0)
        self.assertEqual(self.sub_headings(document, "alpha"), [])
        self.assertIn("Permission is hereby granted, free of charge", document)

    def test_dual_licensed_crate_emits_both_texts_under_labeled_sub_headings(self):
        """`A OR B` is not an election we make for the user, so both
        branches are attributed."""
        self.write_crate(
            "dual",
            "1.0.0",
            files={"LICENSE-MIT": MIT_TEXT, "LICENSE-APACHE": APACHE_TEXT},
        )
        csv_path = self.write_csv(
            [("dual", "https://example.invalid/dual", "MIT OR Apache-2.0", "D Author")]
        )
        lock_path = self.write_lock([("dual", "1.0.0", REGISTRY_SOURCE)])

        code, document, _ = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 0)
        self.assertEqual(
            self.sub_headings(document, "dual"), ["LICENSE-APACHE", "LICENSE-MIT"]
        )
        section = self.section(document, "dual")
        self.assertIn("Permission is hereby granted, free of charge", section)
        self.assertIn("Licensed under the Apache License, Version 2.0", section)

    def test_section_license_line_is_the_declared_expression_not_a_detected_id(self):
        """With every branch's text emitted, the CSV's SPDX expression is
        the accurate label — a single id guessed from the text would
        under-report a dual license."""
        self.write_crate(
            "dual",
            "1.0.0",
            files={"LICENSE-MIT": MIT_TEXT, "LICENSE-APACHE": APACHE_TEXT},
        )
        csv_path = self.write_csv(
            [("dual", "https://example.invalid/dual", "MIT OR Apache-2.0", "D Author")]
        )
        lock_path = self.write_lock([("dual", "1.0.0", REGISTRY_SOURCE)])

        _, document, _ = self.run_generator(csv_path, lock_path)

        self.assertIn("\n## dual\n\nMIT OR Apache-2.0\n", document)

    def test_tarball_text_is_emitted_verbatim_without_placeholder_substitution(self):
        """The other half of "normalization is the only transformation".

        `fill_copyright_placeholders` belongs to the SPDX fallback tier
        alone. If it ever ran on tarball text it would rewrite the Apache
        appendix's `[name of copyright owner]` with the CSV holder — ~99
        sections of the committed artifact carry that literal, and nothing
        else in this suite would fail.
        """
        holder = "A Author <a@example.invalid>"
        self.write_crate("apache-crate", "1.0.0", files={"LICENSE-APACHE": APACHE_APPENDIX_TEXT})
        csv_path = self.write_csv(
            [("apache-crate", "https://example.invalid/apache", "Apache-2.0", holder)]
        )
        lock_path = self.write_lock([("apache-crate", "1.0.0", REGISTRY_SOURCE)])

        code, document, _ = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 0)
        section = self.section(document, "apache-crate")
        self.assertIn("Copyright [yyyy] [name of copyright owner]", section)
        self.assertNotIn(holder, section)

    def test_crlf_shipping_crate_emits_lf_only_output(self):
        self.write_crate(
            "crlf-crate",
            "1.0.0",
            files={"LICENSE": b"MIT License\r\n\r\nCopyright (c) 2015 Someone\r\n"},
        )
        csv_path = self.write_csv(
            [("crlf-crate", "https://example.invalid/crlf", "MIT", "C Author")]
        )
        lock_path = self.write_lock([("crlf-crate", "1.0.0", REGISTRY_SOURCE)])

        code, document, _ = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 0)
        self.assertNotIn("\r", document)

    def test_duplicate_csv_rows_for_one_component_emit_one_section(self):
        """`dd-rust-license-tool` emits a row per copyright holder, so one
        crate can appear several times. Text is keyed off `Cargo.lock`
        now, so the extra rows would be byte-identical sections."""
        self.write_crate("alpha", "1.0.0", files={"LICENSE": MIT_TEXT})
        csv_path = self.write_csv(
            [
                ("alpha", "https://example.invalid/alpha", "MIT", "First Holder"),
                ("alpha", "https://example.invalid/alpha", "MIT", "Second Holder"),
            ]
        )
        lock_path = self.write_lock([("alpha", "1.0.0", REGISTRY_SOURCE)])

        code, document, _ = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 0)
        self.assertEqual(document.count("\n## alpha\n"), 1)


class FencedEmissionTests(_GeneratorHarness):
    """License text is emitted inside a backtick fence.

    Shipped license prose is not inert Markdown: several crates ship
    `#`-prefixed lines that would render as headings and break the
    `## <component>` / `### <file>` structure the document is navigated by
    — and that this suite's own scanner reads. The fence is a container,
    so the bytes between the fences are still the shipped bytes.
    """

    HEADING_LICENSE = (
        "# Some Vendor License\n"
        "\n"
        "## Section 1: Grant\n"
        "\n"
        "### 1.1 Terms\n"
        "\n"
        "Permission is hereby granted, free of charge, to any person.\n"
    )

    def arrange(self, text):
        self.write_crate("fenced", "1.0.0", files={"LICENSE": text})
        csv_path = self.write_csv([("fenced", "https://example.invalid/fenced", "MIT", "F Author")])
        lock_path = self.write_lock([("fenced", "1.0.0", REGISTRY_SOURCE)])
        return csv_path, lock_path

    def test_license_text_is_wrapped_in_a_three_backtick_fence(self):
        code, document, _ = self.run_generator(*self.arrange(MIT_TEXT))

        self.assertEqual(code, 0)
        self.assertIn("\n```\nMIT License", document)
        _, _, unclosed = scan_document(document)
        self.assertIsNone(unclosed, "every fence must close")

    def test_headings_inside_license_text_are_not_document_headings(self):
        """The structural claim: a `## `-prefixed line shipped inside a
        license must not become a section of our document."""
        code, document, _ = self.run_generator(*self.arrange(self.HEADING_LICENSE))

        self.assertEqual(code, 0)
        headings, fenced_headings, _ = scan_document(document)
        self.assertEqual(headings, ["## fenced"])
        self.assertEqual(fenced_headings, 2, "the shipped `## `/`### ` lines stay inside the fence")
        self.assertIn("## Section 1: Grant", self.section(document, "fenced"))

    def test_fence_grows_past_a_backtick_run_inside_the_text(self):
        """A license quoting a fenced code block would otherwise close our
        fence early and spill the rest of the text into the document."""
        text = "MIT License\n\n```\nexample code\n```\n\nPermission is hereby granted.\n"

        code, document, _ = self.run_generator(*self.arrange(text))

        self.assertEqual(code, 0)
        self.assertIn("\n````\nMIT License", document)
        _, _, unclosed = scan_document(document)
        self.assertIsNone(unclosed)
        self.assertIn("example code", self.section(document, "fenced"))

    def test_fence_width_is_one_longer_than_the_longest_run(self):
        cases = [
            ("no backticks", "plain text", "```"),
            ("inline", "use `cargo` here", "```"),
            ("three", "a\n```\nb", "````"),
            ("four", "a\n````\nb", "`````"),
        ]
        for label, text, expected in cases:
            with self.subTest(text=label):
                self.assertEqual(GEN.code_fence_for(text), expected)


class MultiVersionTests(_GeneratorHarness):
    """Several locked versions of one crate share a single section."""

    def test_differing_texts_are_both_emitted_under_version_annotated_headings(self):
        code, document, _ = self.run_generator(
            *self.arrange_two_versions(
                text_v2=MIT_TEXT.replace("Some Upstream Author", "A Different Author")
            )
        )

        self.assertEqual(code, 0)
        self.assertEqual(
            self.sub_headings(document, "multi"),
            ["LICENSE (v1.0.0)", "LICENSE (v2.0.0)"],
        )
        section = self.section(document, "multi")
        self.assertIn("Copyright (c) 2015 Some Upstream Author", section)
        self.assertIn("Copyright (c) 2015 A Different Author", section)

    def test_identical_texts_across_versions_collapse_to_one_unheaded_block(self):
        code, document, _ = self.run_generator(*self.arrange_two_versions())

        self.assertEqual(code, 0)
        self.assertEqual(self.sub_headings(document, "multi"), [])
        self.assertEqual(
            self.section(document, "multi").count("Copyright (c) 2015 Some Upstream Author"), 1
        )

    def test_version_shipping_an_empty_file_does_not_split_the_block_by_version(self):
        """An empty file is not a differing license. Before the b.sv5
        follow-up fix this emitted version-labelled sub-headings asserting
        the two versions were licensed differently. They are not — there
        is one text."""
        code, document, _ = self.run_generator(*self.arrange_two_versions(text_v2=""))

        self.assertEqual(code, 0)
        self.assertEqual(self.sub_headings(document, "multi"), [])
        self.assertIn("Copyright (c) 2015 Some Upstream Author", self.section(document, "multi"))

    def test_version_headings_are_ordered_numerically_not_lexically(self):
        """`10.0.0` follows `9.0.0`. Asserted through the heading builder
        rather than through `version_sort_key` alone, because it is the
        heading order that shows up as spurious drift when a lexical sort
        reorders two sub-headings on the next release."""
        self.write_crate("multi", "9.0.0", files={"LICENSE": MIT_TEXT})
        self.write_crate(
            "multi",
            "10.0.0",
            files={"LICENSE": MIT_TEXT.replace("Some Upstream Author", "A Later Author")},
        )

        blocks, missing, integrity_errors = GEN.build_license_blocks(
            "multi", self.locked_map("multi", "10.0.0", "9.0.0")
        )

        self.assertEqual((missing, integrity_errors), ([], []))
        self.assertEqual(
            [heading for heading, _ in blocks],
            ["LICENSE (v9.0.0)", "LICENSE (v10.0.0)"],
        )


# ── Unresolvable license text ─────────────────────────────────────────────


class UnresolvedLicenseTextTests(_GeneratorHarness):
    """Three crate shapes leave the tarball tier with nothing to attribute.

    All three converge on one path — SPDX fallback, then a hard failure if
    even that resolves nothing — so they are one matrix rather than three
    near-identical tests. Empty and whitespace-only files join the matrix
    because of the b.sv5 follow-up fix: before it, an empty `LICENSE`
    counted as resolved text and the component emitted a section with a
    blank body at exit 0, indistinguishable from a crate whose license we
    had actually captured.

    A component absent from `Cargo.lock` is deliberately NOT here: it has
    its own bucket and is denied the fallback (`UnlockedComponentTests`).
    """

    SHAPES = ("no_license_file", "empty_file", "whitespace_only_file")

    ORIGIN = "https://example.invalid/hollow"

    def arrange(self, shape):
        """Write the inputs for one shape. Returns `(csv_path, lock_path)`."""
        if shape == "no_license_file":
            self.write_crate("hollow", "1.0.0", extra_members=["Cargo.toml", "src/lib.rs"])
        elif shape == "empty_file":
            self.write_crate("hollow", "1.0.0", files={"LICENSE": ""})
        elif shape == "whitespace_only_file":
            self.write_crate("hollow", "1.0.0", files={"LICENSE": "   \n\t\n \n"})
        else:  # Anti-vacuity: a typo must not silently skip a row.
            raise AssertionError("unknown shape {}".format(shape))
        csv_path = self.write_csv([("hollow", self.ORIGIN, "MIT", "H Author")])
        return csv_path, self.write_lock([("hollow", "1.0.0", REGISTRY_SOURCE)])

    def test_shapes_without_spdx_text_exit_non_zero(self):
        """Nothing in either tier resolved: the artifact must not be
        committed with a placeholder where a license belongs."""
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                self.reset_spdx_probe()
                csv_path, lock_path = self.arrange(shape)

                code, document, stderr = self.run_generator(
                    csv_path, lock_path, output_name="{}.md".format(shape)
                )

                self.assertEqual(code, 1)
                self.assertIn("License text not found for the following packages", stderr)
                self.assertIn("License text not found. See: {}".format(self.ORIGIN), document)

    def test_shapes_with_spdx_text_fall_back_to_the_canonical_template(self):
        """The declared expression's canonical text is what gets
        attributed, and the fallback is actually consulted (asserted on
        the fetch log, not just on the emitted text)."""
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                self.reset_spdx_probe()
                self.spdx_texts["MIT"] = SPDX_MIT_TEMPLATE
                csv_path, lock_path = self.arrange(shape)

                code, document, _ = self.run_generator(
                    csv_path, lock_path, output_name="{}.md".format(shape)
                )

                self.assertEqual(code, 0)
                self.assertIn("Copyright (c) <year> H Author", self.section(document, "hollow"))
                self.assertEqual(self.fetched, [GEN.SPDX_URLS["MIT"]])

    def test_empty_and_whitespace_only_texts_are_dropped_before_grouping(self):
        """Asserted on `build_license_blocks` directly, so the claim is
        about the drop itself rather than about whichever tier runs next."""
        cases = [("empty", ""), ("spaces_and_tabs", "   \n\t\n \r\n"), ("newlines", "\n\n\n")]
        for label, content in cases:
            with self.subTest(text=label):
                self.write_crate("blankish", "1.0.0", files={"LICENSE": content})

                self.assertEqual(
                    GEN.build_license_blocks("blankish", self.locked_map("blankish", "1.0.0")),
                    ([], [], []),
                )


class UnlockedComponentTests(_GeneratorHarness):
    """A CSV component with no registry-sourced package in `Cargo.lock`.

    A git or path dependency ships no `.crate`, so its license text cannot
    be read from what we redistribute. That is its own bucket with its own
    operator action (vendor the text, or drop the row), and it is
    deliberately denied the SPDX fallback: boilerplate there would
    attribute a license nobody verified.
    """

    ORIGIN = "https://example.invalid/ghost"

    def arrange(self, locked_packages=()):
        csv_path = self.write_csv([("ghost", self.ORIGIN, "MIT", "G Author")])
        return csv_path, self.write_lock(list(locked_packages))

    def test_component_absent_from_the_lock_exits_in_its_own_bucket(self):
        code, _, stderr = self.run_generator(*self.arrange())

        self.assertEqual(code, 1)
        self.assertIn("CSV components with no registry-sourced package in Cargo.lock", stderr)
        self.assertIn("ghost (MIT)", stderr)

    def test_unlocked_component_is_not_also_reported_as_missing_license_text(self):
        """One cause, one report: the operator action is about the missing
        lock entry, not about a license text we failed to read."""
        code, _, stderr = self.run_generator(*self.arrange())

        self.assertEqual(code, 1)
        self.assertNotIn("License text not found for the following packages", stderr)

    def test_unlocked_component_does_not_take_the_spdx_fallback(self):
        """Even with a template available, no canonical text is fetched or
        emitted for it."""
        self.spdx_texts["MIT"] = SPDX_MIT_TEMPLATE

        code, document, _ = self.run_generator(*self.arrange())

        self.assertEqual(code, 1)
        self.assertEqual(self.fetched, [])
        self.assertIn("License text not found. See: {}".format(self.ORIGIN), document)

    def test_git_sourced_package_counts_as_unlocked(self):
        """Present in the lockfile, but with no `.crate` behind it."""
        code, _, stderr = self.run_generator(
            *self.arrange([("ghost", "0.1.0", "git+https://example.invalid/ghost?rev=abc#abc")])
        )

        self.assertEqual(code, 1)
        self.assertIn("CSV components with no registry-sourced package in Cargo.lock", stderr)


# ── SPDX fallback tier ────────────────────────────────────────────────────


class SpdxFallbackTests(_GeneratorHarness):
    """The tier for the handful of crates that ship no license text.

    Which components reach this tier is `UnresolvedLicenseTextTests`; what
    the tier does once reached is here. Canonical text is emitted for
    *every* branch of the declared expression we have a template for —
    electing one branch put a single license's text under a section
    labelled with the whole expression.
    """

    def arrange(self, license_expr, copyright_info="B Author"):
        self.write_crate("bare", "1.0.0", extra_members=["Cargo.toml"])
        csv_path = self.write_csv(
            [("bare", "https://example.invalid/bare", license_expr, copyright_info)]
        )
        return csv_path, self.write_lock([("bare", "1.0.0", REGISTRY_SOURCE)])

    def test_every_declared_branch_with_a_template_is_emitted(self):
        self.spdx_texts["MIT"] = SPDX_MIT_TEMPLATE
        self.spdx_texts["Apache-2.0"] = SPDX_APACHE_TEMPLATE

        code, document, _ = self.run_generator(*self.arrange("MIT OR Apache-2.0"))

        self.assertEqual(code, 0)
        self.assertEqual(
            self.sub_headings(document, "bare"),
            ["MIT (SPDX canonical text)", "Apache-2.0 (SPDX canonical text)"],
        )
        section = self.section(document, "bare")
        self.assertIn("Permission is hereby granted", section)
        self.assertIn("Apache License", section)

    def test_single_resolved_branch_is_still_provenance_marked(self):
        """The "one block needs no sub-heading" rule does NOT apply here.

        That rule is about shipped files, where a bare block is
        unambiguous. On this tier a bare block would render boilerplate we
        fetched from SPDX identically to a `LICENSE` the crate actually
        shipped — and that distinction is the whole point of the fallback.
        """
        self.spdx_texts["MIT"] = SPDX_MIT_TEMPLATE

        code, document, _ = self.run_generator(*self.arrange("MIT"))

        self.assertEqual(code, 0)
        self.assertEqual(self.sub_headings(document, "bare"), ["MIT (SPDX canonical text)"])
        self.assertIn("Permission is hereby granted", self.section(document, "bare"))

    def test_branch_without_a_template_is_skipped_rather_than_failing(self):
        """`MIT OR Apache-2.0 OR LGPL-2.1-or-later` has no LGPL template.
        The section shows what it has; the run fails only if no branch
        resolved at all."""
        self.spdx_texts["MIT"] = SPDX_MIT_TEMPLATE

        code, document, _ = self.run_generator(
            *self.arrange("MIT OR Apache-2.0 OR LGPL-2.1-or-later")
        )

        self.assertEqual(code, 0)
        self.assertEqual(self.sub_headings(document, "bare"), ["MIT (SPDX canonical text)"])
        self.assertIn("Permission is hereby granted", self.section(document, "bare"))

    def test_duplicate_branch_ids_are_emitted_once(self):
        """`A WITH exception OR A OR B` flattens to a repeated id."""
        self.spdx_texts["MIT"] = SPDX_MIT_TEMPLATE
        self.spdx_texts["Apache-2.0"] = SPDX_APACHE_TEMPLATE

        _, document, _ = self.run_generator(
            *self.arrange("Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT")
        )

        self.assertEqual(
            self.sub_headings(document, "bare"),
            ["Apache-2.0 (SPDX canonical text)", "MIT (SPDX canonical text)"],
        )

    def test_no_resolvable_branch_is_a_hard_failure(self):
        code, _, stderr = self.run_generator(*self.arrange("NoSuchLicense-9.9"))

        self.assertEqual(code, 1)
        self.assertIn("License text not found for the following packages", stderr)

    def test_canonical_text_is_fetched_once_per_id(self):
        """Single-flighted per id: the cache check and the fetch used to be
        separate, so every worker that missed the cache issued its own
        request for the same URL. Asserted single-threaded — the lock is
        the implementation, "one request per id" is the contract."""
        self.spdx_texts["MIT"] = SPDX_MIT_TEMPLATE

        first = GEN.fetch_spdx_license("MIT")
        second = GEN.fetch_spdx_license("MIT")

        self.assertEqual(first, second)
        self.assertEqual(self.fetched, [GEN.SPDX_URLS["MIT"]])

    def test_crate_shipping_license_text_never_reaches_the_network(self):
        """The tarball is the whole input for a crate that ships a license
        file — that is the b.tcm fix. A fetch here would reintroduce the
        upstream-HEAD dependency."""
        self.write_crate("alpha", "1.0.0", files={"LICENSE": MIT_TEXT})
        csv_path = self.write_csv([("alpha", "https://example.invalid/alpha", "MIT", "A Author")])
        lock_path = self.write_lock([("alpha", "1.0.0", REGISTRY_SOURCE)])

        self.run_generator(csv_path, lock_path)

        self.assertEqual(self.fetched, [])

    def test_holder_placeholder_is_filled_and_year_placeholder_is_not(self):
        filled = GEN.fill_copyright_placeholders(
            "Copyright (c) <year> <copyright holders>\n", "A Author <a@example.invalid>"
        )

        self.assertEqual(filled, "Copyright (c) <year> A Author <a@example.invalid>\n")

    def test_holder_containing_regex_escapes_is_inserted_literally(self):
        r"""The holder is CSV data, not a replacement template. Passing it
        as one made a backslash raise `re.error` — swallowed upstream into
        a misleading "License text not found" — and would have let a `\1`
        substitute captured text into a license notice."""
        holder = r"O\Brien \1 <o@example.invalid>"

        filled = GEN.fill_copyright_placeholders(
            "Copyright (c) <year> <copyright holders>\n", holder
        )

        self.assertEqual(filled, "Copyright (c) <year> {}\n".format(holder))

    def test_holder_containing_regex_escapes_survives_a_full_run(self):
        self.spdx_texts["MIT"] = SPDX_MIT_TEMPLATE
        holder = r"O\Brien \1 <o@example.invalid>"

        code, document, _ = self.run_generator(*self.arrange("MIT", copyright_info=holder))

        self.assertEqual(code, 0)
        self.assertIn("Copyright (c) <year> {}".format(holder), self.section(document, "bare"))

    def test_spdx_template_year_placeholder_survives_into_the_output(self):
        """No wall-clock year is synthesized anywhere. The pre-b.sv5
        generator substituted `time.strftime("%Y")` here, which made the
        artifact both wrong and non-deterministic across a new year."""
        self.spdx_texts["MIT"] = SPDX_MIT_TEMPLATE

        _, document, _ = self.run_generator(*self.arrange("MIT"))

        self.assertIn("Copyright (c) <year> B Author", document)
        self.assertNotIn(time.strftime("%Y"), document)


# ── Atomic write ──────────────────────────────────────────────────────────


class AtomicWriteTests(_GeneratorHarness):
    """The document is written beside the target and moved into place only
    once every gate has passed.

    Writing the target directly meant a failed run — a skipped `cargo fetch
    --locked`, blocked egress on the SPDX fetches — replaced the good
    committed `ATTRIBUTIONS.md` with stub text on its way to exiting 1,
    which is exactly when the committed copy matters most.
    """

    COMMITTED = "# Attributions\n\n## previously-committed\n\nMIT\n"

    def arrange_failing(self):
        """One locked crate whose artifact was never fetched."""
        csv_path = self.write_csv([("alpha", "https://example.invalid/alpha", "MIT", "A Author")])
        return csv_path, self.write_lock([("alpha", "1.0.0", REGISTRY_SOURCE)])

    def arrange_passing(self):
        self.write_crate("alpha", "1.0.0", files={"LICENSE": MIT_TEXT})
        csv_path = self.write_csv([("alpha", "https://example.invalid/alpha", "MIT", "A Author")])
        return csv_path, self.write_lock([("alpha", "1.0.0", REGISTRY_SOURCE)])

    def test_failed_run_leaves_the_committed_document_byte_untouched(self):
        csv_path, lock_path = self.arrange_failing()
        target = self.tmpdir / "ATTRIBUTIONS.md"
        target.write_text(self.COMMITTED, encoding="utf-8")

        code, _, _ = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 1)
        self.assertEqual(target.read_bytes().decode("utf-8"), self.COMMITTED)

    def test_failed_run_leaves_the_regenerated_document_in_a_tmp_sibling(self):
        csv_path, lock_path = self.arrange_failing()

        code, _, _ = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 1)
        self.assertTrue(self.tmp_path.exists(), "the .tmp document must be kept for inspection")
        self.assertIn("## alpha", self.tmp_path.read_bytes().decode("utf-8"))

    def test_failure_message_names_the_tmp_path_and_the_untouched_target(self):
        csv_path, lock_path = self.arrange_failing()

        _, _, stderr = self.run_generator(csv_path, lock_path)

        self.assertIn("{} was NOT modified".format(self.target_path), stderr)
        self.assertIn(str(self.tmp_path), stderr)

    def test_successful_run_replaces_the_target_and_leaves_no_tmp(self):
        csv_path, lock_path = self.arrange_passing()
        target = self.tmpdir / "ATTRIBUTIONS.md"
        target.write_text(self.COMMITTED, encoding="utf-8")

        code, document, _ = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 0)
        self.assertFalse(self.tmp_path.exists(), "a successful run must not leave a .tmp behind")
        self.assertNotIn("previously-committed", document)
        self.assertIn("## alpha", target.read_bytes().decode("utf-8"))


# ── Failure gates ─────────────────────────────────────────────────────────


class FailureGateTests(_GeneratorHarness):
    """Every condition that must fail the `supply-chain:attributions-drift`
    job rather than emit a quietly-wrong artifact."""

    def test_locked_crate_absent_from_the_cache_exits_non_zero(self):
        # No `write_crate` call: the version is locked but never fetched.
        csv_path = self.write_csv([("alpha", "https://example.invalid/alpha", "MIT", "A Author")])
        lock_path = self.write_lock([("alpha", "1.0.0", REGISTRY_SOURCE)])

        code, _, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 1)
        self.assertIn("cargo fetch --locked", stderr)

    def test_one_uncached_version_fails_the_run_but_keeps_the_cached_text(self):
        """`missing_artifacts` and `blocks` are not mutually exclusive: a
        partly-fetched cache must still fail the run, and the version it
        did read must still be attributed — a cold cache is a CI
        environment problem, not a reason to silently narrow the
        artifact."""
        code, document, stderr = self.run_generator(
            *self.arrange_two_versions(cached=("1.0.0",))
        )

        self.assertEqual(code, 1)
        self.assertIn("cargo fetch --locked", stderr)
        self.assertIn("multi: 2.0.0", stderr)
        self.assertIn("Copyright (c) 2015 Some Upstream Author", self.section(document, "multi"))

    def test_declared_copyleft_expression_exits_non_zero(self):
        """The primary arm of the permissive gate: the CSV expression
        itself is outside `PERMISSIVE_LICENSES`, so no text inspection is
        needed to reject it."""
        self.write_crate("copyleft", "1.0.0", files={"LICENSE": GPL_TEXT})
        csv_path = self.write_csv(
            [("copyleft", "https://example.invalid/copyleft", "GPL-3.0", "C Author")]
        )
        lock_path = self.write_lock([("copyleft", "1.0.0", REGISTRY_SOURCE)])

        code, _, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 1)
        self.assertIn("Non-permissive licenses detected", stderr)
        self.assertIn("copyleft: GPL-3.0", stderr)

    def test_copyleft_text_under_a_permissive_declaration_exits_non_zero(self):
        """The safety net: the CSV expression can say MIT while the file
        the crate actually ships is GPL. `deny.toml` only checks the
        declaration, so this is the only place the shipped text is read."""
        self.write_crate("sneaky", "1.0.0", files={"LICENSE": GPL_TEXT})
        csv_path = self.write_csv([("sneaky", "https://example.invalid/sneaky", "MIT", "S Author")])
        lock_path = self.write_lock([("sneaky", "1.0.0", REGISTRY_SOURCE)])

        code, _, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 1)
        self.assertIn("Non-permissive licenses detected", stderr)
        self.assertIn("shipped text detected as GPL-3.0", stderr)

    def test_permissive_text_merely_mentioning_the_gpl_passes_the_gate(self):
        """The copyleft signatures require the two-line heading a GNU
        license opens with. A linking exception, or an MPL "Secondary
        License" clause, *names* the GPL in prose — matching those reported
        permissive crates as copyleft and failed the job."""
        mentions_gpl = MIT_TEXT + (
            "\nAs a special exception, linking this library with code covered by the\n"
            "GNU General Public License, Version 3, does not by itself cause the\n"
            "resulting work to be covered by the GNU General Public License.\n"
        )
        self.write_crate("mentions", "1.0.0", files={"LICENSE": mentions_gpl})
        csv_path = self.write_csv(
            [("mentions", "https://example.invalid/mentions", "MIT", "M Author")]
        )
        lock_path = self.write_lock([("mentions", "1.0.0", REGISTRY_SOURCE)])

        code, _, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 0)
        self.assertNotIn("Non-permissive", stderr)

    def test_permissive_declaration_with_matching_text_passes_the_gate(self):
        """Anti-vacuity for the checks above: the same code path must not
        fail an ordinary MIT crate."""
        self.write_crate("alpha", "1.0.0", files={"LICENSE": MIT_TEXT})
        csv_path = self.write_csv([("alpha", "https://example.invalid/alpha", "MIT", "A Author")])
        lock_path = self.write_lock([("alpha", "1.0.0", REGISTRY_SOURCE)])

        code, _, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 0)
        self.assertNotIn("Non-permissive", stderr)

    def test_unreadable_crate_archive_exits_non_zero(self):
        """A truncated `.crate` whose bytes still match the lockfile — the
        tarball itself is unreadable. The `Error processing entry` line is
        what distinguishes this from an ordinary unresolvable component:
        `main`'s blanket handler, not the fallback tier, produced the empty
        block list."""
        self.write_raw_archive("corrupt", "1.0.0", b"not a gzip stream")
        csv_path = self.write_csv(
            [("corrupt", "https://example.invalid/corrupt", "MIT", "C Author")]
        )
        lock_path = self.write_lock([("corrupt", "1.0.0", REGISTRY_SOURCE)])

        code, _, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 1)
        self.assertIn("Error processing entry", stderr)
        self.assertIn("License text not found for the following packages", stderr)


# ── Determinism ───────────────────────────────────────────────────────────


class DeterminismTests(_GeneratorHarness):
    """The drift job diffs a regeneration byte-for-byte against the
    committed file, so the generator must be a pure function of its
    inputs — including across the thread pool that processes entries."""

    def arrange(self):
        self.spdx_texts["MIT"] = SPDX_MIT_TEMPLATE
        self.write_crate("alpha", "1.0.0", files={"LICENSE": MIT_TEXT})
        self.write_crate(
            "dual", "1.0.0", files={"LICENSE-MIT": MIT_TEXT, "LICENSE-APACHE": APACHE_TEXT}
        )
        self.write_crate("multi", "1.0.0", files={"LICENSE": MIT_TEXT})
        self.write_crate(
            "multi",
            "2.0.0",
            files={"LICENSE": MIT_TEXT.replace("Some Upstream Author", "Another Author")},
        )
        self.write_crate("bare", "1.0.0", extra_members=["Cargo.toml"])
        csv_path = self.write_csv(
            [
                ("alpha", "https://example.invalid/alpha", "MIT", "A Author"),
                ("dual", "https://example.invalid/dual", "MIT OR Apache-2.0", "D Author"),
                ("multi", "https://example.invalid/multi", "MIT", "M Author"),
                ("bare", "https://example.invalid/bare", "MIT", "B Author"),
            ]
        )
        lock_path = self.write_lock(
            [
                ("alpha", "1.0.0", REGISTRY_SOURCE),
                ("dual", "1.0.0", REGISTRY_SOURCE),
                ("multi", "1.0.0", REGISTRY_SOURCE),
                ("multi", "2.0.0", REGISTRY_SOURCE),
                ("bare", "1.0.0", REGISTRY_SOURCE),
            ]
        )
        return csv_path, lock_path

    def test_two_runs_on_one_input_are_byte_identical(self):
        csv_path, lock_path = self.arrange()

        first_code, first, _ = self.run_generator(csv_path, lock_path, output_name="first.md")
        second_code, second, _ = self.run_generator(csv_path, lock_path, output_name="second.md")

        self.assertEqual((first_code, second_code), (0, 0))
        self.assertEqual(first, second)

    def test_components_are_emitted_in_csv_order(self):
        """The thread pool completes entries out of order; the section
        order must follow the CSV, not completion timing."""
        for name in ("zulu", "alpha", "mike"):
            self.write_crate(name, "1.0.0", files={"LICENSE": MIT_TEXT})
        csv_path = self.write_csv(
            [(name, "https://example.invalid/{}".format(name), "MIT", "An Author")
             for name in ("zulu", "alpha", "mike")]
        )
        lock_path = self.write_lock(
            [(name, "1.0.0", REGISTRY_SOURCE) for name in ("zulu", "alpha", "mike")]
        )

        code, document, _ = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 0)
        headings, _, _ = scan_document(document)
        self.assertEqual(headings, ["## zulu", "## alpha", "## mike"])


# ── Cargo.lock parsing ────────────────────────────────────────────────────


class ParseCargoLockTests(_GeneratorHarness):
    """`parse_cargo_lock` decides which `(name, version, checksum)` triples
    are looked up in the registry cache. A key that bleeds across a table
    boundary invents a package and turns the run into a hard failure."""

    CHECKSUM_A = "a" * 64
    CHECKSUM_B = "b" * 64

    def test_registry_packages_are_collected_with_versions_and_checksums(self):
        lock_path = self.write_lock(
            [("alpha", "1.0.0", REGISTRY_SOURCE), ("beta", "2.3.4", REGISTRY_SOURCE)],
            checksums={("alpha", "1.0.0"): self.CHECKSUM_A, ("beta", "2.3.4"): self.CHECKSUM_B},
        )

        self.assertEqual(
            GEN.parse_cargo_lock(lock_path),
            {"alpha": {"1.0.0": self.CHECKSUM_A}, "beta": {"2.3.4": self.CHECKSUM_B}},
        )

    def test_missing_checksum_line_yields_none_rather_than_dropping_the_package(self):
        """`None` is an integrity failure downstream, not a skip — a
        silently dropped package would be attributed nothing at all."""
        lock_path = self.write_lock(
            [("alpha", "1.0.0", REGISTRY_SOURCE)], checksums={("alpha", "1.0.0"): OMIT}
        )

        self.assertEqual(GEN.parse_cargo_lock(lock_path), {"alpha": {"1.0.0": None}})

    def test_every_locked_version_of_one_name_is_collected(self):
        lock_path = self.write_lock(
            [("multi", "1.0.0", REGISTRY_SOURCE), ("multi", "2.0.0", REGISTRY_SOURCE)],
            checksums={("multi", "1.0.0"): self.CHECKSUM_A, ("multi", "2.0.0"): self.CHECKSUM_B},
        )

        self.assertEqual(
            GEN.parse_cargo_lock(lock_path),
            {"multi": {"1.0.0": self.CHECKSUM_A, "2.0.0": self.CHECKSUM_B}},
        )

    def test_source_less_package_is_skipped(self):
        """Workspace members and path dependencies have no `.crate`
        artifact, so looking one up would be an unfixable hard failure."""
        lock_path = self.write_lock(
            [("alpha", "1.0.0", REGISTRY_SOURCE), ("live-edit", "0.1.0", None)],
            checksums={("alpha", "1.0.0"): self.CHECKSUM_A},
        )

        self.assertEqual(GEN.parse_cargo_lock(lock_path), {"alpha": {"1.0.0": self.CHECKSUM_A}})

    def test_git_sourced_package_is_skipped(self):
        lock_path = self.write_lock(
            [("forked", "0.1.0", "git+https://example.invalid/forked?rev=abc#abc")]
        )

        self.assertEqual(GEN.parse_cargo_lock(lock_path), {})

    def test_patch_unused_table_does_not_bleed_into_the_preceding_package(self):
        lock_path = self.write_lock_text(
            "\n".join(
                [
                    "[[package]]",
                    'name = "workspace-member"',
                    'version = "0.1.0"',
                    "",
                    "[[patch.unused]]",
                    'name = "workspace-member"',
                    'version = "9.9.9"',
                    'source = "{}"'.format(REGISTRY_SOURCE),
                    "",
                ]
            )
        )

        packages = GEN.parse_cargo_lock(lock_path)

        self.assertNotIn("0.1.0", packages.get("workspace-member", {}))

    def test_metadata_table_does_not_bleed_into_the_preceding_package(self):
        lock_path = self.write_lock_text(
            "\n".join(
                [
                    "[[package]]",
                    'name = "workspace-member"',
                    'version = "0.1.0"',
                    "",
                    "[metadata]",
                    'source = "{}"'.format(REGISTRY_SOURCE),
                    "",
                ]
            )
        )

        self.assertEqual(GEN.parse_cargo_lock(lock_path), {})

    def test_a_packages_source_does_not_carry_into_the_next_package(self):
        lock_path = self.write_lock_text(
            "\n".join(
                [
                    "[[package]]",
                    'name = "alpha"',
                    'version = "1.0.0"',
                    'source = "{}"'.format(REGISTRY_SOURCE),
                    'checksum = "{}"'.format(self.CHECKSUM_A),
                    "",
                    "[[package]]",
                    'name = "workspace-member"',
                    'version = "0.1.0"',
                    "",
                ]
            )
        )

        self.assertEqual(GEN.parse_cargo_lock(lock_path), {"alpha": {"1.0.0": self.CHECKSUM_A}})

    def test_a_packages_checksum_does_not_carry_into_the_next_package(self):
        """The checksum is what the artifact is verified against, so one
        carried across a block boundary would reject a good tarball — or,
        worse, accept the wrong one."""
        lock_path = self.write_lock_text(
            "\n".join(
                [
                    "[[package]]",
                    'name = "alpha"',
                    'version = "1.0.0"',
                    'source = "{}"'.format(REGISTRY_SOURCE),
                    'checksum = "{}"'.format(self.CHECKSUM_A),
                    "",
                    "[[package]]",
                    'name = "beta"',
                    'version = "2.0.0"',
                    'source = "{}"'.format(REGISTRY_SOURCE),
                    "",
                ]
            )
        )

        self.assertEqual(
            GEN.parse_cargo_lock(lock_path),
            {"alpha": {"1.0.0": self.CHECKSUM_A}, "beta": {"2.0.0": None}},
        )

    def test_the_repos_own_cargo_lock_parses_to_a_non_empty_package_set(self):
        """Anti-vacuity against the real file: a regex that stopped
        matching would make every test above pass on synthetic input while
        the generator resolved nothing in CI."""
        packages = GEN.parse_cargo_lock(str(CARGO_LOCK_PATH))

        self.assertGreater(len(packages), 100)
        self.assertNotIn("live-edit", packages, "the workspace member must not be looked up")

    def test_every_registry_package_in_the_repos_lock_has_a_checksum(self):
        """A `None` checksum on a real package would hard-fail the run on
        integrity, so this pins that the `checksum` line is read the way
        the real lockfile writes it."""
        packages = GEN.parse_cargo_lock(str(CARGO_LOCK_PATH))

        unchecked = sorted(
            "{} {}".format(name, version)
            for name, versions in packages.items()
            for version, checksum in versions.items()
            if not checksum
        )
        self.assertEqual(unchecked, [])

    def test_the_repos_own_source_less_packages_are_all_attributed(self):
        """Every source-less package in the real lockfile is a member of the
        real workspace or a `VENDORED_CRATES` entry at its locked version, so
        `supply-chain:attributions-drift` passes the unattributed gate. The
        root package is checked first, so a root manifest the reader no
        longer resolves fails on that rather than on a list of our own
        crates reported as unattributed."""
        members = GEN.workspace_member_names(str(CARGO_LOCK_PATH))

        self.assertIn("live-edit", members or set(), "the root package must be read as a member")
        self.assertEqual(GEN.unattributed_path_packages(str(CARGO_LOCK_PATH)), [])


# ── Vendored (patched-in) crates ──────────────────────────────────────────


class VendoredCrateTests(_GeneratorHarness):
    """`VENDORED_CRATES`: a crate the root manifest patches to a vendored
    tree is resolved in `Cargo.lock` with no `source`, has no `.crate` and
    no CSV row, and would silently drop out of the document. While the patch
    is in force its section is read from the tree; once the crate is
    registry-sourced again nothing changes, and source-less at a version the
    entry does not name it fails the run.

    The fixtures use the real `fred` entry, so they exercise the real
    declaration. Its tree is resolved against the directory holding the
    lockfile, which here is the test's temp dir."""

    NAME = "fred"
    #: The license of the patch's additions and changes, as the note states
    #: it (`vendor/fred-10.1.0.SOURCE.md` § License): pinned here, not read
    #: from the entry, so the published sentence cannot change unseen.
    MODIFICATIONS = (
        "NVIDIA's files under `src/live_edit/` are licensed under Apache-2.0, "
        "and NVIDIA's modifications to fred's own files under MIT."
    )

    def setUp(self):
        super().setUp()
        self.spec = GEN.VENDORED_CRATES[self.NAME]
        self.tree = self.tmpdir / self.spec["path"]

    def write_tree(self, files=(("LICENSE-MIT", MIT_TEXT),)):
        self.tree.mkdir(parents=True)
        for relative, text in files:
            (self.tree / relative).write_text(text, encoding="utf-8")

    def registry_crate(self, name, version="1.0.0"):
        """A cached, locked registry crate with an MIT text, and its CSV row."""
        self.write_crate(name, version, files={"LICENSE-MIT": MIT_TEXT})
        return (name, version, REGISTRY_SOURCE), (
            name,
            "https://example.invalid/" + name,
            "MIT",
            "",
        )

    def sections(self, document):
        headings, _, _ = scan_document(document)
        return [line[len("## "):] for line in headings if line.startswith("## ")]

    def test_parse_path_packages_collects_only_source_less_packages(self):
        lock_path = self.write_lock(
            [
                ("alpha", "1.0.0", REGISTRY_SOURCE),
                (self.NAME, self.spec["version"], None),
                (self.NAME, "9.0.0", REGISTRY_SOURCE),
                ("live-edit", "0.1.0", None),
                ("forked", "0.1.0", "git+https://example.invalid/forked?rev=abc#abc"),
            ]
        )

        self.assertEqual(
            GEN.parse_path_packages(lock_path),
            {self.NAME: {self.spec["version"]}, "live-edit": {"0.1.0"}},
        )

    def test_add_vendored_entries_inserts_at_the_by_component_sort_position(self):
        lock_path = self.write_lock([(self.NAME, self.spec["version"], None)])
        for before, expected in (
            ([], [self.NAME]),
            (["anyhow"], ["anyhow", self.NAME]),
            (["anyhow", "futures"], ["anyhow", self.NAME, "futures"]),
            (["futures", "zeta"], [self.NAME, "futures", "zeta"]),
        ):
            with self.subTest(before=before):
                entries = [{"component": name} for name in before]
                GEN.add_vendored_entries(entries, lock_path)
                self.assertEqual([entry["component"] for entry in entries], expected)
                inserted = next(e for e in entries if e["component"] == self.NAME)
                self.assertEqual(
                    (inserted["license"], inserted["origin"]),
                    (self.spec["license"], self.spec["origin"]),
                )
                self.assertEqual(inserted["vendored"]["tree"], str(self.tree))

    def test_patched_crate_without_a_csv_row_gets_a_section_from_its_tree(self):
        anyhow_lock, anyhow_row = self.registry_crate("anyhow")
        futures_lock, futures_row = self.registry_crate("futures", "0.3.0")
        self.write_tree()
        csv_path = self.write_csv([anyhow_row, futures_row])
        lock_path = self.write_lock(
            [anyhow_lock, (self.NAME, self.spec["version"], None), futures_lock]
        )

        code, document, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.sections(document), ["anyhow", self.NAME, "futures"])
        body = self.section(document, self.NAME)
        self.assertIn(self.spec["license"], body)
        self.assertIn(
            "Vendored with local modifications at `{}`".format(self.spec["path"]), body
        )
        self.assertIn("`{}`".format(self.spec["record"]), body)
        self.assertIn(self.MODIFICATIONS, body)
        self.assertIn("Copyright (c) 2015 Some Upstream Author", body)

    def test_patched_crate_with_a_csv_row_keeps_the_rows_place(self):
        """A CSV row for the crate is kept where the CSV put it — even out
        of sort order — and is read from the tree instead of reported as a
        source-less, unlocked component."""
        anyhow_lock, anyhow_row = self.registry_crate("anyhow")
        zeta_lock, zeta_row = self.registry_crate("zeta")
        self.write_tree()
        fred_row = (self.NAME, self.spec["origin"], self.spec["license"], "")
        csv_path = self.write_csv([anyhow_row, zeta_row, fred_row])
        lock_path = self.write_lock(
            [anyhow_lock, (self.NAME, self.spec["version"], None), zeta_lock]
        )

        code, document, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.sections(document), ["anyhow", "zeta", self.NAME])
        body = self.section(document, self.NAME)
        self.assertIn("Vendored with local modifications", body)
        self.assertIn(self.MODIFICATIONS, body)

    def test_no_section_while_the_patch_is_not_in_force(self):
        """Registry-sourced at the vendored version (the patch was dropped):
        not the vendored crate, and nothing is read from the tree."""
        anyhow_lock, anyhow_row = self.registry_crate("anyhow")
        self.write_tree()
        csv_path = self.write_csv([anyhow_row])
        lock_path = self.write_lock(
            [anyhow_lock, (self.NAME, self.spec["version"], REGISTRY_SOURCE)]
        )

        code, document, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.sections(document), ["anyhow"])

    def test_source_less_crate_at_another_version_fails_and_leaves_the_target(self):
        """Source-less at a version its entry does not name: the crate was
        re-vendored without `VENDORED_CRATES` following. It has no CSV row
        and no section, so the run fails naming both versions rather than
        drop it from the document, and the committed target is untouched."""
        committed = "# Attributions\n\n## previously-committed\n\nMIT\n"
        anyhow_lock, anyhow_row = self.registry_crate("anyhow")
        self.write_tree()
        csv_path = self.write_csv([anyhow_row])
        lock_path = self.write_lock([anyhow_lock, (self.NAME, "99.0.0", None)])
        target = self.tmpdir / "ATTRIBUTIONS.md"
        target.write_text(committed, encoding="utf-8")

        code, document, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 1, stderr)
        self.assertIn(UNATTRIBUTED_PATH_PACKAGES_ERROR, stderr)
        self.assertIn(
            "  - {name} 99.0.0 (VENDORED_CRATES declares {name} {declared})\n".format(
                name=self.NAME, declared=self.spec["version"]
            ),
            stderr,
        )
        self.assertEqual(self.sections(document), ["anyhow"])
        self.assertEqual(target.read_bytes().decode("utf-8"), committed)

    def test_missing_vendored_tree_is_an_integrity_error(self):
        anyhow_lock, anyhow_row = self.registry_crate("anyhow")
        csv_path = self.write_csv([anyhow_row])
        lock_path = self.write_lock([anyhow_lock, (self.NAME, self.spec["version"], None)])

        code, _, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 1)
        self.assertIn("failed verification", stderr)
        self.assertIn(
            "{} {}: vendored tree {} does not exist".format(
                self.NAME, self.spec["version"], self.tree
            ),
            stderr,
        )
        self.assertFalse(self.target_path.exists(), "a failed run must not write the target")

    def test_vendored_tree_without_license_text_is_reported_not_found(self):
        anyhow_lock, anyhow_row = self.registry_crate("anyhow")
        self.write_tree(files=(("README.md", "no license here\n"),))
        csv_path = self.write_csv([anyhow_row])
        lock_path = self.write_lock([anyhow_lock, (self.NAME, self.spec["version"], None)])

        code, _, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 1)
        self.assertIn("License text not found", stderr)
        self.assertIn("  - {} ({})".format(self.NAME, self.spec["license"]), stderr)
        self.assertEqual(self.fetched, [], "a vendored crate never takes the SPDX fallback")


# ── Source-less packages nothing attributes ───────────────────────────────


class UnattributedPathPackageTests(_GeneratorHarness):
    """A source-less `Cargo.lock` package gets no CSV row, so unless it is a
    workspace member (our own code) or a `VENDORED_CRATES` entry at its
    locked version, nothing attributes it, and the run must fail rather
    than leave it out of the document.

    Workspace members are read from the `Cargo.toml` beside the lockfile —
    here the test's temp dir — so each test writes the manifests it needs.
    """

    def write_manifest(self, relative_dir, text):
        """Write `<tmpdir>/<relative_dir>/Cargo.toml` (`"."` is the root)."""
        directory = self.tmpdir / relative_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "Cargo.toml").write_text(text, encoding="utf-8")

    def write_member(self, relative_dir, name):
        self.write_manifest(
            relative_dir, '[package]\nname = "{}"\nversion = "0.1.0"\n'.format(name)
        )

    def members(self):
        """`workspace_member_names` for a lockfile in the temp dir."""
        return GEN.workspace_member_names(self.write_lock([]))

    # ── workspace_member_names ────────────────────────────────────────

    def test_workspace_member_names_expands_member_globs(self):
        self.write_manifest(".", '[workspace]\nmembers = ["crates/*"]\n')
        self.write_member("crates/alpha", "alpha")
        self.write_member("crates/beta", "beta")
        # A matched directory with no manifest is not a member.
        (self.tmpdir / "crates" / "docs").mkdir()

        self.assertEqual(self.members(), {"alpha", "beta"})

    def test_workspace_member_names_excludes_by_path_prefix_not_name_prefix(self):
        """`crates/skip` excludes itself and everything under it, but not
        its sibling `crates/skipper`, whose path merely starts with the
        same characters. Counting `skipper` out would report a real member
        as unattributed; counting `skip/x` in would hide a non-member."""
        self.write_manifest(
            ".",
            '[workspace]\nmembers = ["crates/*", "crates/skip/x"]\n'
            'exclude = ["crates/skip"]\n',
        )
        self.write_member("crates/skip", "skip")
        self.write_member("crates/skip/x", "skip-x")
        self.write_member("crates/skipper", "skipper")

        self.assertEqual(self.members(), {"skipper"})

    def test_workspace_member_names_counts_the_root_package(self):
        """The root `[package]` is a member even when `members` does not
        name `"."`."""
        self.write_manifest(
            ".",
            '[package]\nname = "root-crate"\nversion = "0.1.0"\n\n'
            '[workspace]\nmembers = ["sub"]\n',
        )
        self.write_member("sub", "sub-crate")

        self.assertEqual(self.members(), {"root-crate", "sub-crate"})

    def test_workspace_member_names_without_a_manifest_is_none(self):
        self.assertIsNone(self.members())

    # ── unattributed_path_packages ────────────────────────────────────

    def test_unattributed_path_packages_skips_a_workspace_member(self):
        self.write_manifest(".", '[workspace]\nmembers = ["member"]\n')
        self.write_member("member", "our-crate")
        lock_path = self.write_lock([("our-crate", "0.1.0", None)])

        self.assertEqual(GEN.unattributed_path_packages(lock_path), [])

    def test_unattributed_path_packages_skips_a_vendored_crate_at_its_version(self):
        spec = GEN.VENDORED_CRATES["fred"]
        lock_path = self.write_lock([("fred", spec["version"], None)])

        self.assertEqual(GEN.unattributed_path_packages(lock_path), [])

    def test_unattributed_path_packages_reports_a_vendored_crate_at_another_version(self):
        """Only the version the entry does not name is reported."""
        spec = GEN.VENDORED_CRATES["fred"]
        lock_path = self.write_lock(
            [("fred", spec["version"], None), ("fred", "99.0.0", None)]
        )

        self.assertEqual(GEN.unattributed_path_packages(lock_path), [("fred", "99.0.0")])

    def test_unattributed_path_packages_reports_an_unknown_source_less_package(self):
        """Not a member of the workspace beside the lockfile, and no
        `VENDORED_CRATES` entry: a newly vendored crate."""
        self.write_manifest(".", '[workspace]\nmembers = ["member"]\n')
        self.write_member("member", "our-crate")
        lock_path = self.write_lock(
            [("our-crate", "0.1.0", None), ("stray", "2.0.0", None)]
        )

        self.assertEqual(GEN.unattributed_path_packages(lock_path), [("stray", "2.0.0")])

    # ── main ──────────────────────────────────────────────────────────

    def test_run_with_a_workspace_member_in_the_lock_passes(self):
        """The member gate reaches `main`: our own source-less crates in
        `Cargo.lock` do not fail the run."""
        self.write_crate("anyhow", "1.0.0", files={"LICENSE-MIT": MIT_TEXT})
        csv_path = self.write_csv([("anyhow", "https://example.invalid/anyhow", "MIT", "")])
        self.write_manifest(
            ".", '[package]\nname = "our-crate"\nversion = "0.1.0"\n\n[workspace]\n'
        )
        lock_path = self.write_lock(
            [("anyhow", "1.0.0", REGISTRY_SOURCE), ("our-crate", "0.1.0", None)]
        )

        code, _, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 0, stderr)
        self.assertNotIn(UNATTRIBUTED_PATH_PACKAGES_ERROR, stderr)

    def test_run_with_an_unknown_source_less_package_fails_naming_it(self):
        self.write_crate("anyhow", "1.0.0", files={"LICENSE-MIT": MIT_TEXT})
        csv_path = self.write_csv([("anyhow", "https://example.invalid/anyhow", "MIT", "")])
        lock_path = self.write_lock(
            [("anyhow", "1.0.0", REGISTRY_SOURCE), ("stray", "2.0.0", None)]
        )

        code, _, stderr = self.run_generator(csv_path, lock_path)

        self.assertEqual(code, 1, stderr)
        self.assertIn(UNATTRIBUTED_PATH_PACKAGES_ERROR, stderr)
        self.assertIn("  - stray 2.0.0\n", stderr)
        self.assertFalse(self.target_path.exists(), "a failed run must not write the target")


# ── License-expression grammar ────────────────────────────────────────────


class LicenseExpressionTests(unittest.TestCase):
    """`split_license_expr` / `is_permissive_license` on the expression
    shapes the generated CSV actually carries.

    These decide which SPDX texts a fallback emits and whether the
    permissive gate fires, so the grammar is pinned against real inputs
    rather than invented ones. Every expression below except the ones
    marked synthetic appeared verbatim in a generated
    `LICENSE-3rdparty.csv` when it was written.

    That provenance is a comment, not an assertion, on purpose. The
    dependency set moves under this file — an expression that is real
    today leaves the corpus the moment its last crate is dropped — and a
    test that failed on that would go red for an ordinary `Cargo.lock`
    change with nothing wrong. Every case here earns its place by the
    behaviour it pins; provenance only records *why* that shape was worth
    pinning. Re-check it by generating the CSV and comparing its `License`
    column against these expressions, and demote anything that has left
    the corpus to synthetic rather than deleting it — a shape the
    generator once met is a shape it must still handle.
    """

    #: `(expression, split_ids, permissive)`.
    CASES = [
        # `WITH <exception>` is stripped before splitting, so the exception
        # never becomes a license id of its own.
        (
            "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT",
            ["Apache-2.0", "Apache-2.0", "MIT"],
            True,
        ),
        # Parentheses group: permissive only because the OR is evaluated
        # before the AND.
        ("(MIT OR Apache-2.0) AND Unicode-3.0", ["MIT", "Apache-2.0", "Unicode-3.0"], True),
        # Synthetic: a *parenthesized* WITH, so the exception id abuts the
        # closing paren. Stripping it with `\S+` swallowed the `)` too,
        # leaving an unbalanced expression that parsed as malformed and
        # therefore non-permissive: a wrong hard failure on a permissive
        # crate. `[^\s()]+` is what keeps the group intact. The corpus
        # currently declares this shape unparenthesized (the case above),
        # so this is the regression case for the parenthesized spelling
        # SPDX permits and a future dependency may well ship.
        (
            "(Apache-2.0 WITH LLVM-exception) OR Apache-2.0 OR MIT",
            ["Apache-2.0", "Apache-2.0", "MIT"],
            True,
        ),
        # Synthetic — and necessarily so: the gate rejects copyleft, so a
        # GPL declaration cannot reach the corpus these tests read. It is
        # the *rejection* that has to be exercised on invented input. The
        # exception is dropped, so the branch is judged on its base id:
        # copyleft under OR is still rescued by the permissive sibling...
        ("(GPL-3.0 WITH Classpath-exception-2.0) OR MIT", ["GPL-3.0", "MIT"], True),
        # ...but under AND it is not. Synthetic for the same reason.
        ("(GPL-3.0 WITH Classpath-exception-2.0) AND MIT", ["GPL-3.0", "MIT"], False),
        # AND is a conjunction — every operand must be permissive.
        ("MIT AND BSD-3-Clause", ["MIT", "BSD-3-Clause"], True),
        ("Apache-2.0 AND ISC", ["Apache-2.0", "ISC"], True),
        # Doubled separator whitespace, as the CSV really contains it.
        ("Apache-2.0  OR  MIT", ["Apache-2.0", "MIT"], True),
        # OR is a choice: one permissive branch is enough, and every
        # branch's text is emitted so a reader sees what they are choosing.
        (
            "MIT OR Apache-2.0 OR LGPL-2.1-or-later",
            ["MIT", "Apache-2.0", "LGPL-2.1-or-later"],
            True,
        ),
        # Synthetic: the declared-copyleft gate's input.
        ("GPL-3.0", ["GPL-3.0"], False),
        # Synthetic: a copyleft AND operand is not rescued by a permissive
        # sibling. Flattening the two operators (the previous behaviour)
        # let exactly this through on the strength of the MIT.
        ("MIT AND GPL-3.0", ["MIT", "GPL-3.0"], False),
    ]

    def test_expression_splits_into_its_license_ids(self):
        for expression, expected, _ in self.CASES:
            with self.subTest(expression=expression):
                self.assertEqual(GEN.split_license_expr(expression), expected)

    def test_permissive_verdict_matches_the_deny_toml_allow_list(self):
        for expression, _, expected in self.CASES:
            with self.subTest(expression=expression):
                self.assertEqual(GEN.is_permissive_license(expression), expected)

    def test_expression_shape_predicate_accepts_expressions_and_rejects_prose(self):
        """`looks_like_spdx_expression` is the filter that decides which
        lines of the committed artifact reach the permissive gate, so a
        drift in it silently shrinks that gate's corpus instead of failing.
        Pinned in both directions: every real expression above must pass,
        and the line shapes the scan actually walks past — a URL, a prose
        sentence — must not."""
        for expression, _, _ in self.CASES:
            with self.subTest(expression=expression):
                self.assertTrue(looks_like_spdx_expression(expression))

        for line in [
            "https://github.com/rust-lang/rust/blob/master/LICENSE-MIT",
            "This crate is dual licensed, see the files below for details.",
            "",
        ]:
            with self.subTest(line=line):
                self.assertFalse(looks_like_spdx_expression(line))

    def test_malformed_expression_is_not_permissive(self):
        """This gates a release, so an expression we cannot parse is not
        one we can call permissive."""
        for expression in ["", "(MIT", "MIT OR", "OR MIT", "MIT)", "MIT AND"]:
            with self.subTest(expression=expression):
                self.assertFalse(GEN.is_permissive_license(expression))

    def test_every_real_expression_in_the_generated_csv_is_permissive(self):
        """Anti-vacuity against the real CSV, and a live check of the gate:
        if a dependency bump introduces a non-permissive declaration, this
        fails here rather than only in CI."""
        csv_path = require_license_csv(self)
        with open(csv_path, newline="", encoding="utf-8") as handle:
            expressions = {row["License"] for row in csv.DictReader(handle)}

        self.assertGreater(len(expressions), 5)
        for expression in sorted(expressions):
            with self.subTest(expression=expression):
                self.assertTrue(GEN.is_permissive_license(expression))


# ── Committed artifact ────────────────────────────────────────────────────


class CommittedArtifactTests(unittest.TestCase):
    """Cheap structural checks on the committed `ATTRIBUTIONS.md`.

    Structure, presence and absence only — never absolute counts — so an
    ordinary regeneration after a dependency bump does not fail these. The
    section-list check re-derives its expectation from
    `LICENSE-3rdparty.csv` (plus the `VENDORED_CRATES` in force per
    `Cargo.lock`) rather than hardcoding a number, and so is the
    one case here that goes through `require_license_csv` and skips
    without a generated CSV; everything else reads only the committed
    document and runs in every checkout.
    """

    def setUp(self):
        self.assertTrue(
            ATTRIBUTIONS_PATH.is_file(), "expected the artifact at {}".format(ATTRIBUTIONS_PATH)
        )
        # `read_bytes`, not `read_text`: universal-newline translation would
        # hide a committed CRLF from `test_artifact_is_lf_only`.
        self.text = ATTRIBUTIONS_PATH.read_bytes().decode("utf-8")

    def csv_components(self):
        """Unique components in CSV order — the sections the generator
        emits, by construction."""
        csv_path = require_license_csv(self)
        components = []
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["Component"] not in components:
                    components.append(row["Component"])
        return components

    def test_artifact_starts_with_the_generated_header(self):
        self.assertTrue(
            self.text.startswith("# Attributions\n"),
            "the committed artifact does not start with the header `main()` writes",
        )

    def test_artifact_does_not_carry_the_fabricated_rustls_webpki_copyright(self):
        """The b.sv5 regression itself: a year range whose tail was the
        wall-clock year at generation time, not anything upstream wrote."""
        self.assertNotIn(FABRICATED_COPYRIGHT_LINE, self.text)

    def test_artifact_keeps_apache_appendix_placeholders_verbatim(self):
        """The positive counterpart to the absence checks: the Apache
        appendix's placeholders are boilerplate every Apache-licensed crate
        ships, and a generator that resumed substituting into tarball text
        would quietly rewrite all of them."""
        self.assertIn("Copyright [yyyy] [name of copyright owner]", self.text)

    def test_artifact_is_lf_only(self):
        """CRLF anywhere in the committed file would mean the normalization
        was bypassed, and the drift job would never diff clean again."""
        self.assertNotIn("\r", self.text)

    def test_artifact_has_no_unresolved_license_text(self):
        """`License text not found` is the placeholder `main()` writes for
        a component it could not resolve — the run exits non-zero and no
        longer moves the document into place, so its presence here would
        mean a failed generation was committed by hand."""
        self.assertNotIn("License text not found. See:", self.text)

    def test_artifact_attributes_the_statically_linked_jemalloc_notice(self):
        """jemalloc is compiled into the service binary as its global
        allocator, so its BSD-2-Clause notice must ship, under the crate that
        builds it and with the conjunction declared in `license-tool.toml`."""
        body = section_bodies(self.text).get(EMBEDDING_CRATE)
        self.assertIsNotNone(body, "no `## {}` section".format(EMBEDDING_CRATE))
        headings, _, _ = scan_document(body)

        self.assertIn("### {}".format(EMBEDDED_PATH), headings)
        self.assertIn("Jason Evans", body)
        self.assertEqual(
            next(line for line in body.split("\n") if line.strip()), EMBEDDED_EXPRESSION
        )

    def test_artifact_carries_each_vendored_crates_generated_note(self):
        """Each vendored crate in force carries, under its heading, the note
        the generator writes from its `VENDORED_CRATES` entry and the real
        tree, so a hand edit of the note, or an entry change committed
        without regenerating, fails here without a generated CSV."""
        entries = []
        GEN.add_vendored_entries(entries, str(CARGO_LOCK_PATH))
        self.assertNotEqual(entries, [], "no VENDORED_CRATES entry is in force per Cargo.lock")
        sections = section_bodies(self.text)
        for entry in entries:
            with self.subTest(component=entry["component"]):
                note = GEN.process_vendored_entry(entry, entry["vendored"])["note"]
                body = sections.get(entry["component"])
                self.assertIsNotNone(body, "no `## {}` section".format(entry["component"]))
                # `main()` writes the declared expression, then the note.
                written = [line for line in body.split("\n") if line.strip()][1]
                self.assertEqual(written, note)

    def test_every_fence_in_the_artifact_closes(self):
        """An unbalanced fence means the rest of the document is inside a
        code block — every heading after it stops being a heading."""
        _, _, unclosed = scan_document(self.text)

        self.assertIsNone(unclosed)

    def expected_sections(self):
        """The sections the generator emits: the CSV's components, plus every
        `VENDORED_CRATES` crate whose `[patch.crates-io]` is in force.

        A patched-in crate is resolved in `Cargo.lock` as a source-less
        package at its vendored version, and `dd-rust-license-tool` writes no
        CSV row for a path dependency; the generator then inserts its section
        where the tool's by-component sort would have put the row (or keeps
        the row's place if the CSV does list it). Re-derived here from
        `Cargo.lock` and the CSV, so dropping the patch moves both sides
        together."""
        components = self.csv_components()
        path_packages = GEN.parse_path_packages(str(CARGO_LOCK_PATH))
        for name, spec in GEN.VENDORED_CRATES.items():
            if spec["version"] not in path_packages.get(name, ()) or name in components:
                continue
            position = next(
                (i for i, component in enumerate(components) if component > name),
                len(components),
            )
            components.insert(position, name)
        return components

    def test_artifact_sections_are_exactly_the_csv_components_and_vendored_crates(self):
        """The fence-aware count. A naive `startswith("## ")` scan finds
        more sections than this, because shipped license texts contain
        their own `## ` lines — which is the whole reason the text is
        fenced. Re-derived from the CSV and the vendored crates in force
        (`expected_sections`), so a dependency bump moves both sides
        together."""
        headings, fenced_headings, _ = scan_document(self.text)

        self.assertEqual(
            [line[len("## "):] for line in headings if line.startswith("## ")],
            self.expected_sections(),
        )
        self.assertGreater(
            fenced_headings,
            0,
            "no heading-looking line is fenced — either the corpus changed or the "
            "scan is not seeing inside fences, which would make this check vacuous",
        )

    def section_expressions(self):
        """`{component: declared expression}` from the committed document.

        `main()` writes `## {component}` immediately followed by
        `{license_id}` (generate_attributions.py, the two `output_lines`
        appends at the top of the per-result loop), so the first non-blank
        line of every section body *is* that component's declared SPDX
        expression. Read through the fence-aware `section_bodies` scanner
        for the same reason the heading scan is fence-aware: a `## ` line
        inside a shipped license text is not a section.
        """
        expressions = {}
        for component, body in section_bodies(self.text).items():
            first = next((line for line in body.split("\n") if line.strip()), None)
            self.assertIsNotNone(
                first,
                "section `{}` has an empty body — the generator always writes the "
                "declared expression directly under the heading".format(component),
            )
            expressions[component] = first.strip()
        return expressions

    def test_every_declared_expression_in_the_artifact_is_permissive(self):
        """The permissiveness gate, anchored to the *committed* document
        rather than to the generated CSV.

        The CSV-anchored sibling in `LicenseExpressionTests` skips wherever
        `dd-rust-license-tool write` has not run, which used to mean the
        permissiveness gate had no unconditional anchor at all: on a fresh
        checkout a non-permissive dependency would land with every test
        green. `ATTRIBUTIONS.md` is committed, so this runs everywhere and
        needs nothing generated.

        It does not replace the CSV check. The generator collapses repeated
        rows for one component to a single section, so a row-level
        divergence between two rows of the same crate is visible in the CSV
        and invisible here.
        """
        expressions = self.section_expressions()

        # Anti-vacuity: a scanner that stopped matching sections, or a
        # corpus that collapsed to one license, would make the loop below
        # assert nothing. The document carries 22 distinct expressions
        # across 277 sections today; the floor is deliberately far under
        # that so a dependency bump never trips it.
        self.assertGreater(len(set(expressions.values())), 5)

        # Extraction honesty. `split_license_expr` splits on operators and
        # strips `WITH` exceptions but validates nothing, so it happily
        # returns `['https://github.com/x']` for a line that is not an
        # expression at all — and `is_permissive_license` would then report
        # that as merely "not permissive", pointing the reader at the
        # dependency instead of at the layout change that actually broke
        # this. Every operand must look like an SPDX id.
        unparsed = sorted(
            "{} declares {!r}".format(component, expression)
            for component, expression in expressions.items()
            if not looks_like_spdx_expression(expression)
        )
        self.assertEqual(
            unparsed,
            [],
            "a section's first body line does not read as an SPDX expression. The "
            "generator writes the heading and then the declared expression on the "
            "next line, so this means that layout changed and the permissiveness "
            "gate below is reading the wrong line entirely",
        )

        non_permissive = sorted(
            "{} declares {!r}".format(component, expression)
            for component, expression in expressions.items()
            if not GEN.is_permissive_license(expression)
        )
        self.assertEqual(
            non_permissive,
            [],
            "the committed artifact attributes components under expressions the "
            "permissive gate rejects — the generator would exit non-zero on this "
            "input, so this document could not have been produced by a passing run",
        )


# ── CSV gate ──────────────────────────────────────────────────────────────


class _GateProbe(unittest.TestCase):
    """Stand-in `TestCase` for driving `require_license_csv`'s branches.

    `fail` and `skipTest` are `TestCase` methods that raise, so the only
    way to observe them without failing or skipping the *real* test is to
    hand the helper a throwaway instance and catch what it raises.

    The single method is deliberately named `_unused` rather than
    `runTest`: `TestLoader.loadTestsFromTestCase` falls back to `runTest`
    when a `TestCase` subclass exposes no `test*` methods, so a class
    carrying one would be *collected and executed* by the very `python3 -m
    unittest` invocation this module is run under. With neither a `test*`
    method nor a `runTest`, the loader yields an empty suite for this
    class and it exists only as a construction target below.
    """

    #: The method name every construction passes, so the mismatch between
    #: the name and `TestCase.__init__`'s default can't drift.
    PROBE_METHOD = "_unused"

    def _unused(self):  # pragma: no cover — constructed, never invoked
        raise AssertionError("_GateProbe is a fixture, not a test")

    @classmethod
    def make(cls):
        """A probe instance. `TestCase.__init__` resolves the method name
        eagerly, so it must name something that exists."""
        return cls(cls.PROBE_METHOD)


class RequireLicenseCsvTests(unittest.TestCase):
    """The three branches of `require_license_csv`, hermetically.

    The gate decides whether a missing generated file is a skip or a hard
    failure, and it is the *polarity* that matters: inverted, either the
    strict CI job silently skips its own gates, or every fresh checkout
    fails on a file that is absent by design. Neither shows up in a normal
    run — the branch not taken is invisible — so both are driven here
    against a patched `CSV_PATH` and a patched environment, with no real
    CSV involved either way.

    `CSV_PATH` is patched on the module object rather than by dotted name
    so this works whether the module was imported as
    `tools.tests.test_generate_attributions` or run directly as
    `__main__`.
    """

    @property
    def module(self):
        """This module, resolved at call time rather than in the class
        body — a body-time `sys.modules[__name__]` is an import-time hard
        failure under any loader that registers the module late."""
        return sys.modules[__name__]

    @contextlib.contextmanager
    def csv_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "LICENSE-3rdparty.csv"
            path.write_text("Component,Origin,License,Copyright\n", encoding="utf-8")
            with mock.patch.object(self.module, "CSV_PATH", path):
                yield path

    @contextlib.contextmanager
    def csv_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "LICENSE-3rdparty.csv"
            # Never created: `is_file()` is False without needing a delete.
            with mock.patch.object(self.module, "CSV_PATH", path):
                yield path

    @contextlib.contextmanager
    def require_csv(self, enabled):
        """Pin `REQUIRE_CSV_ENV` regardless of the ambient environment.

        These cases must hold under *both* polarities of the real run —
        `supply-chain:attributions-drift` executes this module with the
        variable already set to `1` — so neither branch may inherit it.
        """
        with mock.patch.dict(os.environ):
            if enabled:
                os.environ[REQUIRE_CSV_ENV] = "1"
            else:
                os.environ.pop(REQUIRE_CSV_ENV, None)
            yield

    def test_present_csv_returns_the_path(self):
        for strict in (False, True):
            with self.subTest(strict=strict):
                with self.csv_present() as path, self.require_csv(strict):
                    self.assertEqual(require_license_csv(_GateProbe.make()), path)

    def test_absent_csv_under_strict_env_fails_naming_the_generating_job(self):
        with self.csv_absent(), self.require_csv(True):
            with self.assertRaises(_GateProbe.failureException) as caught:
                require_license_csv(_GateProbe.make())

        message = str(caught.exception)
        self.assertIn(CI_STRICT_CSV_JOB, message)
        self.assertIn(REQUIRE_CSV_ENV, message)

    def test_absent_csv_without_strict_env_skips_naming_the_generator(self):
        with self.csv_absent(), self.require_csv(False):
            with self.assertRaises(unittest.SkipTest) as caught:
                require_license_csv(_GateProbe.make())

        reason = str(caught.exception)
        self.assertIn(CSV_GENERATOR_CMD, reason)
        self.assertIn(CI_STRICT_CSV_JOB, reason)

    def test_any_non_empty_env_value_arms_strict(self):
        """Presence arms the gate; the value is never parsed.

        The polarity argument is in `strict_csv_required`'s docstring, and
        it is asymmetric: an exact-`1` reading sends `=true` and `=yes`
        down the *skip* branch, so a job that asked for enforcement gets
        none and nothing anywhere says so. `=0` arming strictly is the
        cost of that, and it is the cheap direction — a loud failure
        naming the variable, against a silent hole in a supply-chain gate.

        `0` is in the armed list on purpose: it is not an oversight to be
        "fixed" later by special-casing it back to a skip.
        """
        for value in ["1", "true", "yes", "0"]:
            with self.subTest(value=value, expected="armed"):
                with self.csv_absent(), mock.patch.dict(
                    os.environ, {REQUIRE_CSV_ENV: value}
                ):
                    self.assertTrue(strict_csv_required())
                    with self.assertRaises(_GateProbe.failureException):
                        require_license_csv(_GateProbe.make())

        # Empty and unset are the *same* state to `os.environ.get(…, "")`,
        # and both must reach the skip: an exported-but-empty variable is
        # how a shell spells "not set" often enough that treating it as
        # armed would fail fresh checkouts under a stray `export`.
        with self.csv_absent(), mock.patch.dict(os.environ, {REQUIRE_CSV_ENV: ""}):
            with self.subTest(value="", expected="lenient"):
                self.assertFalse(strict_csv_required())
                with self.assertRaises(unittest.SkipTest):
                    require_license_csv(_GateProbe.make())

        with self.csv_absent(), self.require_csv(False):
            with self.subTest(value=None, expected="lenient"):
                self.assertFalse(strict_csv_required())
                with self.assertRaises(unittest.SkipTest):
                    require_license_csv(_GateProbe.make())

    def test_strict_failure_message_says_how_to_get_the_skip_back(self):
        """The value is not parsed, so an operator who wrote `=0` meaning
        "off" needs the message to tell them that unsetting is the way —
        otherwise the gate is merely surprising instead of fail-closed."""
        with self.csv_absent(), mock.patch.dict(os.environ, {REQUIRE_CSV_ENV: "0"}):
            with self.assertRaises(_GateProbe.failureException) as caught:
                require_license_csv(_GateProbe.make())

        self.assertIn("unset the variable", str(caught.exception))



# ── Strict-mode skip guard ────────────────────────────────────────────────


class StrictModeSkipGuardTests(unittest.TestCase):
    """In a strict run, *no* test in this module may skip — by any route.

    `require_license_csv` converts its own skip into a failure when the
    strict env is armed, but it only governs cases that call it. A future
    CSV-anchored case written with the obvious bare
    `@unittest.skipUnless(CSV_PATH.is_file(), …)` skips in
    `supply-chain:python-tests` *and* in `supply-chain:attributions-drift`,
    asserts nothing in either, and is invisible in both jobs' exit status.
    No per-case review catches that reliably, and nothing else in this file
    can see it at all.

    So the property is asserted one level up, over the whole module: when
    the strict tier runs, its result must carry zero skips, whatever
    produced them. That leaves routing through `require_license_csv` as the
    only way to write a case that reads the CSV and still passes CI.

    Cost is one extra in-process pass over the module (~0.1 s), and the
    precondition is that every case here be re-runnable in the same
    process. They are: fixtures are per-test `tempfile` trees, `CARGO_HOME`
    / `fetch_url` / `registry_cache_dirs`' memo are patched and restored
    per test, and the working-tree tiers (`ATTRIBUTIONS.md`, `Cargo.lock`,
    the CSV) open files read-only.

    **Skips only.** The nested run's failures and errors are deliberately
    not asserted on. In a strict run *without* the CSV the gate turns the
    CSV-anchored cases into failures — which is the correct outcome, is
    already reported against those cases by the outer run, and would
    otherwise be re-reported here as a second, misleading defect pointing
    at the guard. `result.skipped == []` still holds in that scenario, and
    that is right: the guard's question is "did anything escape the gate",
    not "did the suite pass".
    """

    #: Anti-vacuity floor for the collected suite. Loader drift, or a
    #: filter that matched too much, would otherwise leave an empty suite
    #: whose "no skips" is trivially true. Far below the real count so an
    #: ordinary added or removed case never trips it.
    MIN_COLLECTED_TESTS = 40

    @classmethod
    def flatten(cls, suite):
        """Leaf `TestCase`s of a possibly-nested suite."""
        for item in suite:
            if isinstance(item, unittest.TestSuite):
                yield from cls.flatten(item)
            else:
                yield item

    def module_suite(self):
        """Every test in this module except this class's own.

        Excluded by class, not by test id: it is this class that recurses.
        Left in, each nested run would collect and run the guard again,
        which would collect and run it again, unboundedly.

        `sys.modules[__name__]` resolves at call time so this works both
        under `python3 -m unittest tools.tests.…` and under a direct
        `python3 tools/tests/…` run, where the module is `__main__`.
        """
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
        return [
            test
            for test in self.flatten(suite)
            if not isinstance(test, StrictModeSkipGuardTests)
        ]

    def skipped_in(self, tests):
        """`[(test id, reason)]` for everything that skipped in a nested run.

        Output is swallowed. A nested failure is reported by the outer run
        of that same case anyway, and letting its stream writes through
        would interleave a second, unattributed report into the middle of
        the real one.
        """
        result = unittest.TestResult()
        muted = io.StringIO()
        with contextlib.redirect_stdout(muted), contextlib.redirect_stderr(muted):
            unittest.TestSuite(tests).run(result)
        return [(test.id(), reason) for test, reason in result.skipped]

    def test_strict_mode_permits_no_skips(self):
        if not strict_csv_required():
            self.skipTest(
                "this guard only applies to the strict tier: without {} set, the "
                "CSV-anchored cases skip by design and a lenient run legitimately "
                "reports skips. `{}` sets it, and that is where this enforces.".format(
                    REQUIRE_CSV_ENV, CI_STRICT_CSV_JOB
                )
            )

        tests = self.module_suite()
        self.assertGreater(
            len(tests),
            self.MIN_COLLECTED_TESTS,
            "collected only {} tests from this module — the loader is not seeing "
            "the suite, so `no skips` below would pass without checking "
            "anything".format(len(tests)),
        )

        skipped = self.skipped_in(tests)
        self.assertEqual(
            skipped,
            [],
            "{} is armed, but {} test(s) still skipped:\n  {}\nUnder the strict "
            "tier a skip is a silent hole: the case asserts nothing here and "
            "nothing in `supply-chain:python-tests` either. Route any read of a "
            "generated file through `require_license_csv` instead of skipping on "
            "its own condition.".format(
                REQUIRE_CSV_ENV,
                len(skipped),
                "\n  ".join(
                    "{} — {}".format(test_id, reason) for test_id, reason in skipped
                ),
            ),
        )

    def test_the_guard_sees_a_skip(self):
        """Polarity, hermetically and in every environment.

        `assertEqual(skipped, [])` is a gate only while `skipped_in`
        actually reports skips — a helper that returned `[]` unconditionally
        (or a nested run that never executed) passes the guard on a suite
        full of skips and reintroduces exactly the hole this class exists
        to close. Driven on a local probe class so it needs neither the
        strict env nor the CSV, and so the loader never collects the
        deliberately-skipping case as a test of its own.
        """

        class _SkipProbe(unittest.TestCase):
            def test_passes(self):
                pass

            def test_skips(self):
                self.skipTest("deliberate probe skip")

        tests = list(self.flatten(unittest.TestLoader().loadTestsFromTestCase(_SkipProbe)))
        self.assertEqual(len(tests), 2)

        skipped = self.skipped_in(tests)

        self.assertEqual(len(skipped), 1)
        self.assertTrue(
            skipped[0][0].endswith("test_skips"),
            "expected the skipping probe case, got {}".format(skipped[0][0]),
        )
        self.assertEqual(skipped[0][1], "deliberate probe skip")

    def test_collected_suite_excludes_only_this_class(self):
        """Both directions of the recursion filter. Too narrow and the
        guard recurses into itself; too wide and it silently stops watching
        the classes that can actually skip."""
        collected = {type(test).__name__ for test in self.module_suite()}

        self.assertNotIn(type(self).__name__, collected)
        for expected in ["RequireLicenseCsvTests", "CommittedArtifactTests"]:
            with self.subTest(cls=expected):
                self.assertIn(expected, collected)


if __name__ == "__main__":
    unittest.main()
