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

"""Generate ATTRIBUTIONS.md from LICENSE-3rdparty.csv + the locked crate artifacts.

Input is the `dd-rust-license-tool write` CSV (component / origin / license /
copyright) plus `Cargo.lock`. License *text* is read from the locally cached
`.crate` tarball for each `(name, version)` pinned in `Cargo.lock` — never from
a project's GitHub default branch.

Why the tarball: the tarball is the artifact we redistribute, so the `LICENSE*`
files inside it are by definition the text we owe attribution for. Sourcing the
text from upstream HEAD instead made the output depend on third-party commit
activity (unreleased copyright edits landed in our ATTRIBUTIONS.md before they
were in any version we ship, turning the CI drift job red with no local change).
The `.crate` cache is populated by `cargo fetch --locked`.

Emission rules
--------------
* Every top-level license file a crate ships is emitted — `LICEN[CS]E*`,
  `COPYING*`, `UNLICENSE*`, `COPYRIGHT*` (case-insensitive), matched against the
  tarball's actual members rather than guessed from a filename list.
* A license file nested below the crate root is emitted only when declared in
  `EMBEDDED_LICENSE_FILES`: a crate that compiles an embedded library into our
  binary (tikv-jemalloc-sys builds jemalloc's C sources) owes that library's
  notice too, and it lives in the embedded tree, not at the crate root. It is
  emitted under a `### <path>` sub-heading naming its path in the crate. A
  declared file missing from the verified tarball fails the run.
* Dual-licensed crates (`A OR B`) emit **every** file under a `### <filename>`
  sub-heading. We do not elect a license branch on the user's behalf, so the
  attribution carries both texts. A crate shipping exactly one license file is
  emitted without a sub-heading.
* Every license text is emitted inside a fenced code block, with a fence one
  backtick longer than the longest backtick run in the text. License prose is
  full of Markdown-active characters — several crates ship `#`-prefixed lines
  that would otherwise become document headings and corrupt the structure the
  `## <component>` / `### <file>` headings define. Fencing is a container, not a
  transformation: the bytes between the fences are the shipped bytes.
* When several versions of one crate are locked and a given license file's text
  differs between them, each distinct text is emitted and its sub-heading names
  the versions it came from.
* The section's license line is the SPDX expression declared in the CSV (after
  `license-tool.toml` overrides), not a single id guessed from the text — with
  every branch's text emitted, the expression is the accurate label.
* License text is **never** rewritten to synthesize a copyright year. Only the
  SPDX canonical-text fallback (see below) gets holder-name placeholders filled;
  its `<year>` / `[yyyy]` placeholders are deliberately left intact rather than
  fabricated from the wall clock.
* Line endings are normalized to LF. A couple of crates ship CRLF license files;
  git normalizes the committed file to LF on checkout, so emitting CRLF would
  make the drift job diff a LF working copy against a CRLF regeneration and fail
  forever. This is the only transformation applied to shipped text.

Vendored crates
---------------
A crate the root `Cargo.toml` patches to a vendored tree (`VENDORED_CRATES`)
has no `.crate` artifact and no CSV row. While `Cargo.lock` resolves it as a
source-less package at the vendored version, its section is read from the
vendored tree with the same file selection, and carries a line naming the tree
and the record of its patch, and stating the license of the patch's additions
and modifications. Any other source-less package that is not a
workspace member — a crate re-vendored at a version its entry does not name, or
vendored with no entry — fails the run instead of silently going unattributed.

Fallback
--------
A handful of crates ship no license file at all. Those fall back to the SPDX
canonical text, fetched from a **pinned** `license-list-data` release tag
(`SPDX_TAG`) — the only remaining network input. Canonical text is emitted for
*every* branch of the declared expression that has a known template, not for one
elected branch: the section's label is the whole expression, so showing one
branch's text under it misrepresents the other branches. Branches with no known
template are simply absent; the run fails only if no branch resolved at all.

Integrity
---------
The `(name, version)` pin is not by itself an identity: more than one registry
can be configured (crates.io plus a mirror), and each may hold a
`<name>-<version>.crate`. The chosen artifact is therefore SHA-256-verified
against the `checksum` recorded in `Cargo.lock` before its bytes are read.

Exit status
-----------
Non-zero if any component ends up with no license text, if a CSV component has
no registry-sourced package in `Cargo.lock`, if a source-less `Cargo.lock`
package is neither a workspace member nor a `VENDORED_CRATES` entry at its
locked version, if a non-permissive (GPL-family)
license is detected, if a locked crate's `.crate` tarball is absent from the
local registry cache, if a cached tarball fails its checksum, or if a verified
tarball lacks a file `EMBEDDED_LICENSE_FILES` declares for it.

The output file is written to `<output>.tmp` first and moved into place only
after every gate passes, so a failed run never overwrites a good committed
`ATTRIBUTIONS.md`. A failed run deliberately leaves `<output>.tmp` behind for
inspection.
"""

import csv
import functools
import glob
import hashlib
import os
import re
import socket
import sys
import tarfile
import time
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# Top-level tarball members treated as license text. Matched case-insensitively
# against the member's basename; the tarball is enumerated rather than probed
# for a fixed filename list, so crate-specific names (LICENSE.httprouter,
# LICENSE-BoringSSL, license-apache-2.0, ...) are picked up automatically.
LICENSE_FILE_RE = re.compile(r"^(LICEN[CS]E|COPYING|UNLICENSE|COPYRIGHT)", re.IGNORECASE)
# License files nested below a crate root that must be emitted as well, keyed by
# crate name, as paths relative to the crate root. Only the root is enumerated
# automatically, because a nested license file usually belongs to code the crate
# carries but does not build into our binary. The entries here are the
# exceptions that are built in:
#   * tikv-jemalloc-sys compiles jemalloc's C sources into the service binary
#     (its global allocator). jemalloc is BSD-2-Clause, which requires its
#     notice to be reproduced, and the crate root ships only the binding's own
#     MIT / Apache-2.0 texts. `license-tool.toml` declares the matching
#     `(MIT OR Apache-2.0) AND BSD-2-Clause` expression.
# Each declared path must be present in every locked version's verified tarball;
# a missing one fails the run instead of silently dropping the notice.
EMBEDDED_LICENSE_FILES = {
    "tikv-jemalloc-sys": ("jemalloc/COPYING",),
}
# Third-party crates this repository vendors with a local patch, keyed by crate
# name, with paths relative to the directory holding `Cargo.lock`. While the
# root `Cargo.toml`'s `[patch.crates-io]` points a crate at its vendored tree,
# `Cargo.lock` records it with no `source` or `checksum`, `dd-rust-license-tool`
# leaves it out of the CSV (it lists only packages that have a source), and
# there is no registry `.crate` artifact to read. Its section is built from the
# vendored tree instead — the copy we compile and redistribute — under the
# declared license, with a line saying it is a modified copy. That line ends
# with the entry's `modifications` sentence: the license of what the patch
# adds and changes, as the `record` states it.
#
# The entry applies only while `Cargo.lock` resolves the crate at `version` as a
# source-less package. Once the patch is dropped the crate is registry-sourced
# again and is attributed from its `.crate` like any other. A source-less crate
# with no entry at its locked version (re-vendored without updating `version`,
# or newly vendored) would have no section, so `unattributed_path_packages`
# fails the run on it rather than let it drop out silently.
VENDORED_CRATES = {
    "fred": {
        "version": "10.1.0",
        "path": "vendor/fred-10.1.0",
        "license": "MIT",
        "origin": "https://github.com/aembke/fred.rs",
        "record": "vendor/fred-10.1.0.SOURCE.md",
        "modifications": (
            "NVIDIA's files under `src/live_edit/` are licensed under Apache-2.0, "
            "and NVIDIA's modifications to fred's own files under MIT."
        ),
    },
}
# Largest crate license file we will decompress from a `.crate` tarball. Real
# license texts are tens of KiB at most; the cap only exists to bound memory.
MAX_LICENSE_FILE_BYTES = 1024 * 1024

# Pinned SPDX license-list-data release. Bump deliberately; `main` is an
# unpinned upstream input and this file is asserted byte-for-byte by the
# `supply-chain:attributions-drift` CI job.
SPDX_TAG = "v3.28.0"

SPDX_LICENSE_IDS = [
    "MIT",
    "Apache-2.0",
    "ISC",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "Unlicense",
    "MPL-2.0",
    "Zlib",
    "0BSD",
    "CC0-1.0",
    "BSL-1.0",
    "Unicode-3.0",
]

SPDX_URLS = {
    license_id: f"https://raw.githubusercontent.com/spdx/license-list-data/{SPDX_TAG}/text/{license_id}.txt"
    for license_id in SPDX_LICENSE_IDS
}

# Kept in step with the `[licenses] allow` list in deny.toml.
PERMISSIVE_LICENSES = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unlicense",
    "Zlib",
    "0BSD",
    "CC0-1.0",
    "BSL-1.0",
    "MIT-0",
    "Unicode-3.0",
    "MPL-2.0",
    "WTFPL",
    "Public Domain",
    "bzip2-1.0.6",
    "CDLA-Permissive-2.0",
}

# Holder-name placeholders only. There is deliberately no year placeholder here:
# the wall-clock year is not the crate's copyright year, and inventing one
# produced a notice that was both wrong and non-deterministic across runs.
COPYRIGHT_HOLDER_PATTERNS = [
    r"<copyright holders>",
    r"\[fullname\]",
    r"<COPYRIGHT HOLDER>",
    r"\[name of copyright owner\]",
    r"<owner>",
]

#: Flags every signature is matched under. `MULTILINE` is what makes the
#: copyleft `^` anchors below mean "start of a line" rather than "start of the
#: file"; no other signature uses `^` or `$`.
SIGNATURE_FLAGS = re.IGNORECASE | re.DOTALL | re.MULTILINE


def copyleft_title_signature(title, version):
    """Regex matching a copyleft license *title* as a two-line heading.

    Detecting copyleft hard-fails a release-gating job, so these signatures have
    to fire on text that IS the license, not on text that mentions one. The
    `<phrase>.*<phrase>` shape the permissive signatures use cannot do that:
    under `DOTALL` it pairs a title in one paragraph with a version string
    anywhere later in the file. Permissive text is full of that shape — a
    `COPYRIGHT` block naming the GPL in a linking exception or a "no GPL code
    is included" statement, MPL-2.0 §1.12 ("Secondary License" names the GPL,
    LGPL and AGPL) — and the gate reported those crates as GPL.

    So require the heading GNU license texts actually open with: the title
    alone on its line (indentation and Markdown decoration allowed), then the
    version at the start of the next line. A prose reference — "...covered by
    the GNU General Public License, Version 3, and to distribute" — fails on
    both counts.
    """
    return (
        r"^[ \t#*>]*"
        + title
        + r"[ \t,]*\r?\n(?:[ \t]*\r?\n)?[ \t#*>=_-]*"
        + version
        + r"\b"
    )


LICENSE_SIGNATURES = [
    ("MIT", [
        r"Permission is hereby granted, free of charge",
        r"MIT License",
        r"THE SOFTWARE IS PROVIDED \"AS IS\".*WITHOUT WARRANTY OF ANY KIND",
    ]),
    ("Apache-2.0", [
        r"Apache License.*Version 2\.0",
        r"Licensed under the Apache License, Version 2\.0",
        r"http://www\.apache\.org/licenses/LICENSE-2\.0",
    ]),
    ("BSD-3-Clause", [
        r"Redistribution and use in source and binary forms.*with or without.*modification.*are permitted.*Neither the name of",
    ]),
    ("BSD-2-Clause", [
        r"Redistribution and use in source and binary forms.*with or without.*modification.*are permitted",
    ]),
    ("ISC", [
        r"ISC License",
        r"Permission to use, copy, modify, and/or distribute this software",
    ]),
    ("MPL-2.0", [
        r"Mozilla Public License.*2\.0",
        r"This Source Code Form is subject to the terms of the Mozilla Public License",
    ]),
    ("Unlicense", [
        r"This is free and unencumbered software released into the public domain",
        r"unlicense\.org",
    ]),
    ("Zlib", [
        r"zlib License",
        r"This software is provided 'as-is', without any express or implied warranty",
    ]),
    ("CC0-1.0", [
        r"CC0 1\.0 Universal",
        r"Creative Commons Zero",
        r"creativecommons\.org/publicdomain/zero",
    ]),
    ("BSL-1.0", [
        r"Boost Software License",
        r"boost\.org/LICENSE_1_0\.txt",
    ]),
    ("0BSD", [
        r"Zero-Clause BSD",
        r"Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted",
    ]),
    # Copyleft signatures are deliberately narrower than the permissive ones
    # above — see `copyleft_title_signature`. The short SPDX id is accepted only
    # inside an `SPDX-License-Identifier:` declaration line; a bare "GPL-3.0" in
    # prose is a mention, not a license. Order matters: LESSER / AFFERO titles
    # do not contain the plain "GNU GENERAL PUBLIC LICENSE" title, so the
    # narrower ids are not shadowed.
    ("GPL-3.0", [
        copyleft_title_signature(r"GNU GENERAL PUBLIC LICENSE", r"Version 3"),
        r"^[ \t#*>/]*SPDX-License-Identifier:[ \t]*GPL-3\.0",
    ]),
    ("GPL-2.0", [
        copyleft_title_signature(r"GNU GENERAL PUBLIC LICENSE", r"Version 2"),
        r"^[ \t#*>/]*SPDX-License-Identifier:[ \t]*GPL-2\.0",
    ]),
    ("LGPL-3.0", [
        copyleft_title_signature(r"GNU LESSER GENERAL PUBLIC LICENSE", r"Version 3"),
        r"^[ \t#*>/]*SPDX-License-Identifier:[ \t]*LGPL-3\.0",
    ]),
    ("LGPL-2.1", [
        copyleft_title_signature(r"GNU LESSER GENERAL PUBLIC LICENSE", r"Version 2\.1"),
        r"^[ \t#*>/]*SPDX-License-Identifier:[ \t]*LGPL-2\.1",
    ]),
    ("AGPL-3.0", [
        copyleft_title_signature(r"GNU AFFERO GENERAL PUBLIC LICENSE", r"Version 3"),
        r"^[ \t#*>/]*SPDX-License-Identifier:[ \t]*AGPL-3\.0",
    ]),
]

COPYLEFT_LICENSE_IDS = {
    "GPL-3.0",
    "GPL-2.0",
    "LGPL-3.0",
    "LGPL-2.1",
    "AGPL-3.0",
}


def normalize_newlines(text):
    """Convert CRLF / lone-CR line endings to LF.

    `ATTRIBUTIONS.md` is committed and then asserted byte-for-byte against a
    fresh regeneration by CI. Git normalizes the checked-out file to LF, so a
    CRLF-shipping crate (generic-array, nu-ansi-term) would otherwise wedge the
    drift job permanently.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def fill_copyright_placeholders(license_text, copyright_info):
    """Substitute the copyright *holder* into an SPDX canonical template.

    Only reachable on the SPDX-fallback tier — text read from a crate tarball is
    the real notice and is emitted verbatim. Year placeholders are left intact
    on purpose: we do not know the crate's copyright year and will not invent
    one from the wall clock.
    """
    if not license_text or not copyright_info:
        return license_text

    for pattern in COPYRIGHT_HOLDER_PATTERNS:
        # The holder is CSV data, not a regex replacement template: a lambda
        # inserts it literally. Passing it as a template made a backslash in a
        # holder name raise `re.error` (swallowed upstream into a misleading
        # "License text not found") and would have let a `\1` corrupt the text.
        license_text = re.sub(
            pattern, lambda _m: copyright_info, license_text, flags=re.IGNORECASE
        )

    return license_text


def split_license_expr(license_expr):
    expr = license_expr.strip()
    # `[^\s()]+`, not `\S+`: an exception id abutting a closing paren —
    # `(Apache-2.0 WITH LLVM-exception) OR MIT` — would otherwise be consumed
    # along with the `)`, leaving the expression unbalanced.
    expr = re.sub(r"\s+WITH\s+[^\s()]+", "", expr)
    expr = expr.replace("(", "").replace(")", "")
    return [lic.strip() for lic in re.split(r"\s+(?:OR|AND)\s+", expr) if lic.strip()]


def detect_license_type(license_text, original_license_expr):
    if not license_text:
        return original_license_expr

    original_licenses = set(split_license_expr(original_license_expr))

    for license_id, patterns in LICENSE_SIGNATURES:
        if license_id not in original_licenses:
            continue
        for pattern in patterns:
            if re.search(pattern, license_text, SIGNATURE_FLAGS):
                return license_id

    for license_id, patterns in LICENSE_SIGNATURES:
        for pattern in patterns:
            if re.search(pattern, license_text, SIGNATURE_FLAGS):
                return license_id

    return original_license_expr


def is_permissive_license(license_expr):
    """Evaluate an SPDX expression against `PERMISSIVE_LICENSES`.

    `OR` is a choice, so one permissive branch is enough — `MIT OR
    Apache-2.0 OR LGPL-2.1-or-later` is permissive because we may take the MIT
    branch. `AND` is a conjunction: every operand's terms bind simultaneously,
    so `MIT AND GPL-3.0` is NOT permissive. Flattening the two (the previous
    behaviour) let a copyleft `AND` operand pass on a permissive sibling.

    `AND` binds tighter than `OR`, and parentheses override — the real CSV
    carries `(MIT OR Apache-2.0) AND Unicode-3.0`, which is permissive only if
    the group is evaluated before the `AND`.

    A malformed expression evaluates to `False`: this gates a release, so an
    expression we cannot read is not one we can call permissive. That makes the
    `WITH`-stripping below load-bearing — see `split_license_expr`: swallowing a
    `)` along with the exception id turned a well-formed expression into an
    unparseable one, and the fail-closed verdict then hard-failed the job on a
    permissive crate.
    """
    expr = re.sub(r"\s+WITH\s+[^\s()]+", "", license_expr.strip(), flags=re.IGNORECASE)
    tokens = re.findall(r"\(|\)|[^\s()]+", expr)
    pos = 0

    def parse_or():
        nonlocal pos
        value = parse_and()
        while pos < len(tokens) and tokens[pos].upper() == "OR":
            pos += 1
            value = parse_and() or value
        return value

    def parse_and():
        nonlocal pos
        value = parse_atom()
        while pos < len(tokens) and tokens[pos].upper() == "AND":
            pos += 1
            value = parse_atom() and value
        return value

    def parse_atom():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("truncated expression")
        token = tokens[pos]
        pos += 1
        if token == "(":
            value = parse_or()
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError("unbalanced parentheses")
            pos += 1
            return value
        if token in (")", "OR", "AND"):
            raise ValueError(f"unexpected token {token!r}")
        return token in PERMISSIVE_LICENSES

    try:
        result = parse_or()
    except ValueError:
        return False
    if pos != len(tokens):
        return False
    return result


MAX_WORKERS = 10
BASE_DELAY = 1.0
MAX_DELAY = 60.0

license_cache = {}
license_fetch_locks = {}
cache_lock = threading.Lock()
rate_limit_lock = threading.Lock()
global_backoff = {"delay": 0.0, "until": 0.0}


def wait_for_rate_limit():
    with rate_limit_lock:
        now = time.time()
        if now < global_backoff["until"]:
            wait_time = global_backoff["until"] - now
            time.sleep(wait_time)


def set_rate_limit_backoff(delay):
    with rate_limit_lock:
        global_backoff["delay"] = min(delay, MAX_DELAY)
        global_backoff["until"] = time.time() + global_backoff["delay"]


def reset_rate_limit():
    with rate_limit_lock:
        global_backoff["delay"] = 0.0
        global_backoff["until"] = 0.0


def fetch_url(url, retries=3):
    for attempt in range(retries):
        wait_for_rate_limit()
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 license-fetcher"})
            with urlopen(req, timeout=15) as response:
                reset_rate_limit()
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as e:
            if e.code == 429:
                retry_after = e.headers.get("Retry-After")
                if retry_after:
                    delay = float(retry_after)
                else:
                    delay = BASE_DELAY * (2**attempt)
                set_rate_limit_backoff(delay)
                print(f"Rate limited, backing off {delay:.1f}s...", file=sys.stderr)
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
            elif e.code == 404:
                return None
            if attempt < retries - 1:
                time.sleep(BASE_DELAY * (attempt + 1))
        except (URLError, TimeoutError, socket.timeout):
            # `URLError` covers connection-refused, DNS failure, etc. The
            # read-timeout from `urlopen(..., timeout=15)` raises
            # `socket.timeout` (aliased to the built-in `TimeoutError` in
            # Python 3.10+), which is NOT a `URLError` subclass — without
            # this branch it propagated to the caller and aborted the whole
            # attributions run on a single transient network blip.
            if attempt < retries - 1:
                time.sleep(BASE_DELAY * (attempt + 1))
    return None


# ── Cargo.lock + registry cache ────────────────────────────────────────────────


def parse_cargo_lock(path):
    """Return {crate name: {version: checksum}} for registry-sourced packages.

    Workspace members and path/git dependencies carry no `source` key (or a
    non-registry one) and have no `.crate` artifact, so they are skipped.

    The `checksum` is carried alongside the version because the version alone
    does not identify an artifact — see `find_crate_archive`. A registry package
    always records one; `None` here is an integrity failure, not a skip.
    """
    packages = {}
    for name, version, source, checksum in locked_packages(path):
        if source and source.startswith("registry+"):
            packages.setdefault(name, {})[version] = checksum
    return packages


def parse_path_packages(path):
    """Return {crate name: {version, ...}} for the packages `Cargo.lock` records
    with no `source` key: workspace members and path dependencies, among them
    each `VENDORED_CRATES` entry while its `[patch.crates-io]` is in force."""
    packages = {}
    for name, version, source, _checksum in locked_packages(path):
        if source is None:
            packages.setdefault(name, set()).add(version)
    return packages


def locked_packages(path):
    """Return `(name, version, source, checksum)` for every `[[package]]` block
    of `Cargo.lock`; `source` / `checksum` are `None` where the block has none."""
    packages = []
    in_package = False
    name = version = source = checksum = None

    def flush():
        if in_package and name and version:
            packages.append((name, version, source, checksum))

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("["):
                # Any table header ends the current block, and only a literal
                # `[[package]]` opens one we collect. Resetting the keys alone
                # is not enough: a `[[patch.unused]]` entry carries the same
                # name / version / source / checksum shape, so without tracking
                # which table is open it flushes as a real package — and a
                # patched-out dependency, whose `.crate` we never fetch because
                # we do not ship it, would hard-fail the run as a missing
                # artifact.
                flush()
                in_package = line.strip() == "[[package]]"
                name = version = source = checksum = None
                continue
            m = re.match(r'^(name|version|source|checksum) = "(.*)"$', line)
            if not m:
                continue
            if m.group(1) == "name":
                name = m.group(2)
            elif m.group(1) == "version":
                version = m.group(2)
            elif m.group(1) == "checksum":
                checksum = m.group(2)
            else:
                source = m.group(2)
    flush()
    return packages


def version_sort_key(version):
    core, _, pre = version.partition("-")
    numbers = []
    for part in core.split("."):
        numbers.append(int(part) if part.isdigit() else 0)
    while len(numbers) < 3:
        numbers.append(0)
    # A release sorts after its own pre-releases.
    return (numbers[0], numbers[1], numbers[2], pre == "", pre)


@functools.lru_cache(maxsize=1)
def registry_cache_dirs():
    """Cache directories to search, resolved on first use rather than at import.

    Resolving at import time baked in whatever `CARGO_HOME` happened to be set
    when the module was loaded, so a caller that set it afterwards was ignored.
    Memoized because it globs the filesystem and every crate looks it up; call
    `registry_cache_dirs.cache_clear()` after changing `CARGO_HOME`.
    """
    cargo_home = os.environ.get("CARGO_HOME") or os.path.join(
        os.path.expanduser("~"), ".cargo"
    )
    return tuple(sorted(glob.glob(os.path.join(cargo_home, "registry", "cache", "*"))))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_crate_archive(name, version, checksum):
    """Return `(path, integrity_error)` for a locked crate's cached `.crate`.

    `(None, None)` means "not cached anywhere" — a different failure (run
    `cargo fetch --locked`), reported separately by the caller.

    More than one registry cache is normal: this container alone has
    `index.crates.io` and an internal mirror, and both may hold a
    `<name>-<version>.crate`. Picking the first hit by filename would let the
    document attribute text from an artifact we do not ship, so every candidate
    is SHA-256-checked against `Cargo.lock` and the first that matches wins. A
    name collision in one registry therefore degrades to "keep looking", while
    no candidate matching is a hard failure rather than a silent substitution.
    """
    if not checksum:
        return None, (
            "no `checksum` recorded in Cargo.lock — the artifact cannot be "
            "verified against the lockfile"
        )

    candidates = [
        candidate
        for candidate in (
            os.path.join(cache_dir, f"{name}-{version}.crate")
            for cache_dir in registry_cache_dirs()
        )
        if os.path.exists(candidate)
    ]
    if not candidates:
        return None, None

    for candidate in candidates:
        if sha256_file(candidate) == checksum:
            return candidate, None

    return None, (
        "cached .crate does not match the Cargo.lock checksum "
        f"{checksum} (checked: {', '.join(candidates)})"
    )


def read_crate_license_files(name, version, checksum):
    """Return `({path: text}, integrity_error)` for one crate tarball.

    The files dict is `None` when the `.crate` artifact is not in the local
    registry cache (caller turns that into a hard failure — `cargo fetch
    --locked` fixes it) or when it failed verification, in which case
    `integrity_error` says why. Both are hard failures; they are kept apart
    because the operator action differs.

    Keys are paths relative to the crate root: a top-level license file's
    basename, or a nested path declared in `EMBEDDED_LICENSE_FILES` for this
    crate. A declared path absent from the verified tarball is reported as an
    integrity error, so the notice cannot be dropped without failing the run.
    """
    archive, integrity_error = find_crate_archive(name, version, checksum)
    if archive is None:
        return None, integrity_error

    embedded = EMBEDDED_LICENSE_FILES.get(name, ())
    files = {}
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            # `<name>-<version>/<path>`; the key is `<path>`.
            parts = member.name.split("/")
            relative = "/".join(parts[1:])
            # Top level of the crate root only, unless declared as embedded.
            if relative not in embedded and (
                len(parts) != 2 or not LICENSE_FILE_RE.match(parts[1])
            ):
                continue
            # Bound the decompressed read: the tarball's checksum is verified
            # against Cargo.lock, but a license file is a few KiB, so anything
            # past the cap is not a license and must not be inflated into
            # memory.
            if member.size > MAX_LICENSE_FILE_BYTES:
                return None, (
                    f"{archive}: member {member.name} is {member.size} bytes, "
                    f"above the {MAX_LICENSE_FILE_BYTES}-byte license-file cap"
                )
            handle = tf.extractfile(member)
            if handle is None:
                continue
            files[relative] = normalize_newlines(
                handle.read().decode("utf-8", errors="replace")
            )
    absent = [path for path in embedded if path not in files]
    if absent:
        return None, (
            f"{archive}: declared embedded license file(s) "
            f"{', '.join(absent)} not in the tarball — the crate's layout changed; "
            "update EMBEDDED_LICENSE_FILES in tools/generate_attributions.py to "
            "the new path of the embedded library's notice"
        )
    return files, None


def read_vendored_license_files(name, tree):
    """Return `({path: text}, error)` for a `VENDORED_CRATES` tree on disk.

    The same selection `read_crate_license_files` applies to a tarball: every
    top-level license file, plus the nested paths `EMBEDDED_LICENSE_FILES`
    declares for the crate. A missing tree or declared file is an error.
    """
    if not os.path.isdir(tree):
        return None, f"vendored tree {tree} does not exist"
    embedded = EMBEDDED_LICENSE_FILES.get(name, ())
    files = {}
    candidates = sorted(
        entry for entry in os.listdir(tree) if LICENSE_FILE_RE.match(entry)
    ) + list(embedded)
    for relative in candidates:
        full = os.path.join(tree, relative)
        if not os.path.isfile(full):
            continue
        if os.path.getsize(full) > MAX_LICENSE_FILE_BYTES:
            return None, (
                f"{full} is above the {MAX_LICENSE_FILE_BYTES}-byte license-file cap"
            )
        with open(full, "rb") as f:
            files[relative] = normalize_newlines(f.read().decode("utf-8", errors="replace"))
    absent = [path for path in embedded if path not in files]
    if absent:
        return None, f"{tree}: declared embedded license file(s) {', '.join(absent)} not present"
    return files, None


# ── SPDX fallback ──────────────────────────────────────────────────────────────


def fetch_spdx_license(license_id):
    """Canonical SPDX text for one id, fetched at most once per process.

    Single-flighted per id: the cache check and the fetch used to be separate,
    so every worker that missed the cache issued its own request for the same
    URL (six requests for two distinct licenses on this workspace). The network
    is the one input that can fail, so each redundant request is a way for a
    deterministic run to go red.
    """
    with cache_lock:
        if license_id in license_cache:
            return license_cache[license_id]
        fetch_lock = license_fetch_locks.setdefault(license_id, threading.Lock())

    with fetch_lock:
        with cache_lock:
            # A racing worker may have populated it while we waited.
            if license_id in license_cache:
                return license_cache[license_id]

        url = SPDX_URLS.get(license_id)
        if not url:
            return None
        content = fetch_url(url)
        if not content:
            return None

        content = normalize_newlines(content)
        with cache_lock:
            license_cache[license_id] = content
        return content


# ── Section assembly ───────────────────────────────────────────────────────────


def code_fence_for(text):
    """Return a backtick fence long enough to contain `text`.

    License prose is not inert Markdown. Shipped texts contain `#`-prefixed
    lines (webpki-roots' CDLA, smartstring's MPL exhibits) that render as
    headings and break the `## <component>` / `### <file>` structure this
    document is navigated by. A fence is a container: it changes how the text is
    *read*, not the bytes emitted, so "CRLF→LF is the only transformation"
    still holds. The fence is one backtick longer than the longest run inside so
    a text containing its own fence cannot terminate ours early.
    """
    longest_run = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest_run + 1)


def build_license_blocks(name, locked):
    """Return `(blocks, missing_versions, integrity_errors)` for one component.

    `locked` is the `{version: checksum}` map from `parse_cargo_lock`. A block is
    `(heading_or_None, text)`, where the heading is the `### ...` sub-heading
    label, or `None` when the component ships exactly one license text and needs
    no sub-heading.

    Empty / whitespace-only license files are dropped: a component whose only
    shipped license files are empty yields no blocks, so the caller treats it
    like a component that ships none at all (SPDX fallback, then hard fail).
    """
    per_version = {}
    missing = []
    integrity_errors = []
    for version in locked:
        files, integrity_error = read_crate_license_files(name, version, locked[version])
        if integrity_error is not None:
            integrity_errors.append((version, integrity_error))
            continue
        if files is None:
            missing.append(version)
            continue
        per_version[version] = files

    if not per_version:
        return [], missing, integrity_errors

    ordered_versions = sorted(per_version, key=version_sort_key)
    filenames = sorted({fn for files in per_version.values() for fn in files})

    blocks = []
    for filename in filenames:
        # Group the versions that ship this file by the exact text they ship.
        groups = {}
        for version in ordered_versions:
            text = per_version[version].get(filename)
            if text is None or not text.strip():
                continue
            groups.setdefault(text, []).append(version)
        if not groups:
            continue
        ambiguous = len(groups) > 1
        for text, group_versions in groups.items():
            if ambiguous:
                label = f"{filename} (v" + ", v".join(group_versions) + ")"
            else:
                label = filename
            blocks.append((label, text))

    if len(blocks) == 1:
        blocks = [(None, blocks[0][1])]

    return blocks, missing, integrity_errors


def spdx_fallback_blocks(license_expr, copyright_info):
    """Canonical SPDX text for every branch of `license_expr` we have a template
    for.

    Electing one branch (the old behaviour) put a single license's text under a
    section labelled with the whole expression — `r-efi` is declared `MIT OR
    Apache-2.0 OR LGPL-2.1-or-later` and showed only MIT. A branch with no
    template contributes nothing; the caller hard-fails only when no branch
    resolved at all, so the section shows what it has rather than nothing.

    Every block gets its `### <id> (SPDX canonical text)` heading, including the
    single-branch case that the "one block needs no sub-heading" rule would
    otherwise leave bare. That rule is about shipped files, where the absence of
    a heading is unambiguous; here it would render boilerplate we fetched from
    SPDX identically to a `LICENSE` the crate actually shipped. The distinction
    is the whole point of the fallback tier and belongs in the document.
    """
    blocks = []
    for license_id in split_license_expr(license_expr):
        if license_id not in SPDX_URLS:
            continue
        if any(heading == license_id for heading, _ in blocks):
            continue
        text = fetch_spdx_license(license_id)
        if text:
            blocks.append((license_id, fill_copyright_placeholders(text, copyright_info)))

    return [(f"{license_id} (SPDX canonical text)", text) for license_id, text in blocks]


def process_entry(entry, index, total, locked_versions):
    component = entry["component"]
    origin = entry["origin"]
    license_expr = entry["license"]
    copyright_info = entry.get("copyright", "")

    print(f"[{index}/{total}] Collecting license text for {component}...", file=sys.stderr)

    vendored = entry.get("vendored")
    if vendored is not None:
        return process_vendored_entry(entry, vendored)

    locked = locked_versions.get(component)
    unlocked = locked is None
    blocks, missing, integrity_errors = build_license_blocks(component, locked or {})

    if not blocks and not missing and not integrity_errors and not unlocked:
        # The crate ships no license file at all — fall back to SPDX canonical
        # text. Deliberately NOT reached for an unlocked component: boilerplate
        # there would paper over a component we never read an artifact for.
        blocks = spdx_fallback_blocks(license_expr, copyright_info)

    return {
        "component": component,
        "license": license_expr,
        "origin": origin,
        "blocks": blocks,
        "missing_artifacts": missing,
        "integrity_errors": integrity_errors,
        "unlocked": unlocked,
    }


def process_vendored_entry(entry, vendored):
    """The section for a `VENDORED_CRATES` crate, read from its vendored tree.

    A missing tree or declared embedded file is reported as an integrity
    error; a tree that ships no license text lands in "License text not
    found". The SPDX fallback is never taken: the tree is the shipped artifact.
    """
    component = entry["component"]
    files, error = read_vendored_license_files(component, vendored["tree"])
    blocks = []
    for filename in sorted(files or {}):
        if files[filename].strip():
            blocks.append((filename, files[filename]))
    if len(blocks) == 1:
        blocks = [(None, blocks[0][1])]
    return {
        "component": component,
        "license": entry["license"],
        "origin": entry["origin"],
        "blocks": blocks,
        "note": (
            f"Vendored with local modifications at `{vendored['path']}`: the published "
            f"{component} {vendored['version']} crate plus the patch recorded in "
            f"`{vendored['record']}`. {vendored['modifications']}"
        ),
        "missing_artifacts": [],
        "integrity_errors": [] if error is None else [(vendored["version"], error)],
        "unlocked": False,
    }


def workspace_member_names(lock_path):
    """Return the package names of the workspace `Cargo.lock` belongs to, or
    `None` when there is no `Cargo.toml` beside it.

    Read from that root manifest: its own `[package]`, and every
    `[workspace] members` entry (glob patterns expanded, `exclude` honoured)
    that holds a `Cargo.toml` with a `[package]` name.

    `exclude` is matched as Cargo matches it, by path prefix: a member at or
    under an excluded directory is not a member. That is the direction that
    matters, because counting a non-member as a member would let
    `unattributed_path_packages` skip it silently. Python's `glob` otherwise
    only approximates Cargo's member globs (no `**` recursion, no hidden
    entries, and a pattern matching nothing is not kept as a literal path);
    every such difference can only leave a real member out, which reports it
    as unattributed and fails the run loudly rather than hiding a package.
    """
    root = os.path.dirname(os.path.abspath(lock_path))
    manifest_path = os.path.join(root, "Cargo.toml")
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, "rb") as f:
        manifest = tomllib.load(f)
    names = set()
    root_name = manifest.get("package", {}).get("name")
    if root_name:
        names.add(root_name)
    workspace = manifest.get("workspace", {})
    excluded = [os.path.normpath(os.path.join(root, p)) for p in workspace.get("exclude", [])]

    def is_excluded(member_dir):
        return any(
            member_dir == ex or member_dir.startswith(ex.rstrip(os.sep) + os.sep)
            for ex in excluded
        )

    for pattern in workspace.get("members", []):
        for member_dir in glob.glob(os.path.join(root, pattern)):
            member_dir = os.path.normpath(member_dir)
            member_manifest = os.path.join(member_dir, "Cargo.toml")
            if is_excluded(member_dir) or not os.path.isfile(member_manifest):
                continue
            with open(member_manifest, "rb") as f:
                member_name = tomllib.load(f).get("package", {}).get("name")
            if member_name:
                names.add(member_name)
    return names


def unattributed_path_packages(lock_path):
    """Return `(name, version)` for every source-less `Cargo.lock` package that
    is neither a workspace member nor a `VENDORED_CRATES` entry at its locked
    version, sorted.

    A source-less package is built from a local path. `dd-rust-license-tool`
    gives it no CSV row, so unless `add_vendored_entries` claims it, nothing
    attributes it. Workspace members are our own code and need no section; any
    other — typically a crate re-vendored at a new version without its
    `VENDORED_CRATES` entry following — would drop out of the document with no
    error, so each one fails the run. The workspace manifest is read only when
    some package is unclaimed by `VENDORED_CRATES`; with no manifest beside the
    lockfile, every such package is reported.
    """
    candidates = []
    for name, versions in parse_path_packages(lock_path).items():
        vendored_version = VENDORED_CRATES.get(name, {}).get("version")
        candidates.extend((name, v) for v in versions if v != vendored_version)
    if not candidates:
        return []
    members = workspace_member_names(lock_path) or set()
    return sorted(
        (name, version) for name, version in candidates if name not in members
    )


def add_vendored_entries(entries, lock_path):
    """Mark or insert the `VENDORED_CRATES` sections `Cargo.lock` calls for.

    A vendored crate counts only while `Cargo.lock` resolves it at the vendored
    version with no `source` (its `[patch.crates-io]` is in force). A CSV row
    for it, if any, keeps its place and is read from the tree; otherwise the
    section is inserted where `dd-rust-license-tool`'s by-component sort would
    have put the row.
    """
    path_packages = parse_path_packages(lock_path)
    lock_dir = os.path.dirname(os.path.abspath(lock_path))
    for name, spec in VENDORED_CRATES.items():
        if spec["version"] not in path_packages.get(name, ()):
            continue
        vendored = {
            "path": spec["path"],
            "version": spec["version"],
            "record": spec["record"],
            "modifications": spec["modifications"],
            "tree": os.path.join(lock_dir, spec["path"]),
        }
        existing = next((entry for entry in entries if entry["component"] == name), None)
        if existing is not None:
            existing["vendored"] = vendored
            continue
        position = next(
            (i for i, entry in enumerate(entries) if entry["component"] > name), len(entries)
        )
        entries.insert(
            position,
            {
                "component": name,
                "origin": spec["origin"],
                "license": spec["license"],
                "copyright": "",
                "vendored": vendored,
            },
        )


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "LICENSE-3rdparty.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "ATTRIBUTIONS.md"
    lock_path = sys.argv[3] if len(sys.argv) > 3 else "Cargo.lock"

    locked_versions = parse_cargo_lock(lock_path)

    entries = []
    seen = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            component = row["Component"]
            # One section per crate. `dd-rust-license-tool` can emit several
            # rows for one crate (different copyright holders across locked
            # versions); the license text is now keyed off Cargo.lock, so those
            # rows would produce byte-identical sections.
            if component in seen:
                continue
            seen.add(component)

            entries.append(
                {
                    "component": component,
                    "origin": row["Origin"],
                    "license": row["License"],
                    "copyright": row.get("Copyright", ""),
                }
            )

    add_vendored_entries(entries, lock_path)
    unattributed = unattributed_path_packages(lock_path)

    total = len(entries)
    results = [None] * total

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_entry, entry, i + 1, total, locked_versions): i
            for i, entry in enumerate(entries)
        }
        for future, idx in futures.items():
            try:
                results[idx] = future.result()
            except Exception as e:  # noqa: BLE001 — reported below, then exits 1
                print(f"Error processing entry {idx}: {e}", file=sys.stderr)
                results[idx] = {
                    "component": entries[idx]["component"],
                    "license": entries[idx]["license"],
                    "origin": entries[idx]["origin"],
                    "blocks": [],
                    "missing_artifacts": [],
                    "integrity_errors": [],
                    "unlocked": False,
                }

    missing_licenses = []
    missing_artifacts = []
    integrity_failures = []
    unlocked_components = []
    non_permissive = []

    output_lines = ["# Attributions\n"]

    for result in results:
        component = result["component"]
        license_id = result["license"]
        origin = result["origin"]
        blocks = result["blocks"]

        if result["missing_artifacts"]:
            missing_artifacts.append((component, result["missing_artifacts"]))

        for version, integrity_error in result["integrity_errors"]:
            integrity_failures.append((component, version, integrity_error))

        if result["unlocked"]:
            unlocked_components.append((component, origin, license_id))
        elif not blocks and not result["missing_artifacts"] and not result["integrity_errors"]:
            # One cause, one report. A component that failed to yield text
            # *because* it is unlocked, uncached or unverifiable is already
            # named in the bucket for that cause; "License text not found" is
            # the symptom, not an independent finding, and repeating it doubles
            # the failure list while pointing at the wrong remedy. This branch
            # is the genuine case: the artifact was read, and it carries no
            # license text and no SPDX template resolved.
            missing_licenses.append((component, origin, license_id))

        if not is_permissive_license(license_id):
            non_permissive.append((component, license_id, origin))
        else:
            # Safety net: the declared expression can be permissive while the
            # text actually shipped is copyleft.
            for _, text in blocks:
                detected = detect_license_type(text, license_id)
                if detected in COPYLEFT_LICENSE_IDS:
                    non_permissive.append(
                        (component, f"{license_id} (shipped text detected as {detected})", origin)
                    )
                    break

        output_lines.append(f"\n## {component}\n")
        output_lines.append(f"\n{license_id}\n")
        if result.get("note"):
            output_lines.append(f"\n{result['note']}\n")

        if blocks:
            for heading, text in blocks:
                if heading:
                    output_lines.append(f"\n### {heading}\n")
                fence = code_fence_for(text)
                output_lines.append(f"\n{fence}\n{text.strip()}\n{fence}\n")
        else:
            output_lines.append(f"\nLicense text not found. See: {origin}\n")

    # Write beside the target and move it into place only once every gate below
    # has passed. Writing the target directly meant a failed run (a skipped
    # `cargo fetch --locked`, blocked egress on the SPDX fetches) replaced the
    # good committed ATTRIBUTIONS.md with stub text on its way to exiting 1.
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(output_lines)

    has_error = False

    if missing_artifacts:
        has_error = True
        print("\n" + "=" * 60, file=sys.stderr)
        print("ERROR: locked crate artifacts missing from the registry cache:", file=sys.stderr)
        print("Run `cargo fetch --locked` and re-run this script.", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for component, versions in missing_artifacts:
            print(f"  - {component}: {', '.join(versions)}", file=sys.stderr)

    if integrity_failures:
        has_error = True
        print("\n" + "=" * 60, file=sys.stderr)
        print("ERROR: cached crate artifacts failed verification:", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for component, version, reason in integrity_failures:
            print(f"  - {component} {version}: {reason}", file=sys.stderr)

    if unlocked_components:
        has_error = True
        print("\n" + "=" * 60, file=sys.stderr)
        print(
            "ERROR: CSV components with no registry-sourced package in Cargo.lock:",
            file=sys.stderr,
        )
        print(
            "A git or path dependency ships no `.crate` artifact, so its license "
            "text cannot be read from what we redistribute. Vendor the text "
            "(a vendored crate belongs in VENDORED_CRATES) or drop the component "
            "from the CSV — do not let it fall back to SPDX boilerplate, which "
            "would attribute a license we never verified.",
            file=sys.stderr,
        )
        print("=" * 60, file=sys.stderr)
        for component, origin, license_id in unlocked_components:
            print(f"  - {component} ({license_id})", file=sys.stderr)
            print(f"    Origin: {origin}", file=sys.stderr)

    if unattributed:
        has_error = True
        print("\n" + "=" * 60, file=sys.stderr)
        print(
            "ERROR: source-less Cargo.lock packages that are neither workspace "
            "members nor VENDORED_CRATES at their locked version:",
            file=sys.stderr,
        )
        print(
            "A package Cargo.lock records with no `source` is built from a local "
            "path, gets no CSV row, and would be left out of the document. If a "
            "crate was re-vendored at a new version, or newly vendored, update or "
            "add its VENDORED_CRATES entry (version, path, license, origin, "
            "record, modifications) to match Cargo.lock. Workspace members are read from the "
            "Cargo.toml beside Cargo.lock.",
            file=sys.stderr,
        )
        print("=" * 60, file=sys.stderr)
        for name, version in unattributed:
            declared = VENDORED_CRATES.get(name, {}).get("version")
            if declared:
                print(
                    f"  - {name} {version} (VENDORED_CRATES declares {name} {declared})",
                    file=sys.stderr,
                )
            else:
                print(f"  - {name} {version}", file=sys.stderr)

    if missing_licenses:
        has_error = True
        print("\n" + "=" * 60, file=sys.stderr)
        print("ERROR: License text not found for the following packages:", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for component, origin, license_id in missing_licenses:
            print(f"  - {component} ({license_id})", file=sys.stderr)
            print(f"    Origin: {origin}", file=sys.stderr)

    if non_permissive:
        has_error = True
        print("\n" + "=" * 60, file=sys.stderr)
        print("ERROR: Non-permissive licenses detected:", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for component, license_id, origin in non_permissive:
            print(f"  - {component}: {license_id}", file=sys.stderr)
            print(f"    Origin: {origin}", file=sys.stderr)

    if has_error:
        print("\n" + "=" * 60, file=sys.stderr)
        if missing_artifacts:
            print(f"Total packages with missing artifacts: {len(missing_artifacts)}", file=sys.stderr)
        if integrity_failures:
            print(f"Total packages failing verification: {len(integrity_failures)}", file=sys.stderr)
        if unlocked_components:
            print(f"Total components absent from Cargo.lock: {len(unlocked_components)}", file=sys.stderr)
        if unattributed:
            print(f"Total unattributed source-less packages: {len(unattributed)}", file=sys.stderr)
        if missing_licenses:
            print(f"Total packages with missing licenses: {len(missing_licenses)}", file=sys.stderr)
        if non_permissive:
            print(f"Total packages with non-permissive licenses: {len(non_permissive)}", file=sys.stderr)
        print(
            f"\n{output_path} was NOT modified. The regenerated document is at "
            f"{tmp_path} for inspection; delete it when you are done.",
            file=sys.stderr,
        )
        print("=" * 60, file=sys.stderr)
        sys.exit(1)

    os.replace(tmp_path, output_path)
    print(f"Generated {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
