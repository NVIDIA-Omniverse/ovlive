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
"""Add and verify the NVIDIA Apache-2.0 license header, and check that every
published file carries a license record.

The header is NVIDIA's standard SPDX-tagged Apache-2.0 notice, verbatim, in the
file's comment style. `check` gives every file one of four outcomes; the rules
and their reasons are in docs-internal/engineering-best-practices.md § License
headers:

- Internal: a file `.publishignore` excludes (`published_paths` mirrors the
  snapshot builder's matching) needs the header only if `requires_header`
  selects it (source code, scripts, Dockerfiles, build manifests, pip
  requirements files, configuration including ignore files, Helm chart files,
  proptest seed files, `.b64` data pools), and needs nothing else.
- Preserved upstream: a file of a vendored tree (`VENDORED_TREES`) listed in
  the tree's upstream-file manifest `<tree>.SHA256SUMS` keeps upstream's
  notice and must not carry NVIDIA's header; if its bytes differ from the
  listed digest, the tree's patch `<tree>.patch` must edit it. Any other file
  in the tree is an NVIDIA addition: the patch must add it, and it follows
  the rules below like any other file.
- Header: a published file `requires_header` selects, and a published
  Markdown file without a classification entry, carries the header.
  Markdown takes it unprefixed inside an HTML comment, `<!--` ... `-->`,
  starting on the first line, so it renders nothing and stays ahead of a
  MyST `(label)=` line.
- Classified: any other published file, which has no comment syntax, has an
  entry in `tools/license-classification.json` (`CLASSES`: the classes and
  the fields each needs). `check` validates every entry.

    python3 tools/license_headers.py check           # CI gate: supply-chain:license-headers
    python3 tools/license_headers.py fix             # add, repair or relicense headers in place
    python3 tools/license_headers.py fix PATH...     # generators: header their own output

Both commands cover tracked files and untracked files git does not ignore, so
`fix` reaches a file the moment it is created. `fix` never headers an internal
document, an upstream file of a vendored tree or a classified file, and
`fix PATH` refuses a published file with no comment syntax and no entry:
classify it instead. `check` also verifies every other file that already carries an
NVIDIA header, so a header can be absent from a file that needs none but never
malformed. The header must lead the file
(only a shebang, an encoding line, parser directives or its comment opener may
precede it). `check` rejects the NVIDIA proprietary header this repository
carried before it became Apache-2.0, and, in the comment block that holds a
header (lines in the header's own comment style; never code), proprietary-notice
wording (also when wrapped across lines) or a second NVIDIA header after it;
another author's copyright and license lines there are allowed. It checks the
header and that block only: it is not a content filter for the rest of a file. `fix` replaces
only the lines that are literally header text, matched line by line, so it
never touches the comments or code below; it replaces the proprietary header,
and refuses (fix by hand) whatever it cannot match exactly: an altered or
truncated body, unknown text under a proprietary identifier, a copyright line
that names anyone but NVIDIA, or proprietary text or another NVIDIA header under
the header. A leading byte-order mark stays first. `fix`
preserves an existing header's copyright year and new headers get the current
year. Standard library only: the CI job runs on `python:3.12-slim`, which has
no git, so the file list falls back to a directory walk there. The publish
snapshot's own check (verify rule R14) runs this module in the snapshot, which
has no `.publishignore`, so every file there counts as published.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

COPYRIGHT_TEMPLATE = (
    "SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved."
)
COPYRIGHT_RE = re.compile(
    r"SPDX-FileCopyrightText: Copyright \(c\) (?P<year>\d{4}(?:-\d{4})?) "
    r"NVIDIA CORPORATION & AFFILIATES\. All rights reserved\."
)
BODY = (
    "SPDX-License-Identifier: Apache-2.0",
    "",
    'Licensed under the Apache License, Version 2.0 (the "License");',
    "you may not use this file except in compliance with the License.",
    "You may obtain a copy of the License at",
    "",
    "    http://www.apache.org/licenses/LICENSE-2.0",
    "",
    "Unless required by applicable law or agreed to in writing, software",
    'distributed under the License is distributed on an "AS IS" BASIS,',
    "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
    "See the License for the specific language governing permissions and",
    "limitations under the License.",
)
HEADER_LINES = 1 + len(BODY)
MARKER = "SPDX-FileCopyrightText"
LICENSE_ID = BODY[0]
# The NVIDIA proprietary header this repository carried before it became
# Apache-2.0, recognised only so `fix` can replace it and `check` can name it.
LEGACY_BODY = (
    "SPDX-License-Identifier: LicenseRef-NvidiaProprietary",
    "",
    "NVIDIA CORPORATION, its affiliates and licensors retain all intellectual",
    "property and proprietary rights in and to this material, related",
    "documentation and any modifications thereto. Any use, reproduction,",
    "disclosure or distribution of this material and related documentation",
    "without an express license agreement from NVIDIA CORPORATION or",
    "its affiliates is strictly prohibited.",
)
LEGACY_LICENSE_ID = LEGACY_BODY[0]
LEGACY_LAST_LINE = LEGACY_BODY[-1]
# A header is found only within this many lines. It must lead the file: only a
# shebang, an encoding declaration, Dockerfile parser directives, blank lines or
# its own comment opener may precede it (`preamble_problem`), which is also
# where `insert` puts a new one.
SEARCH_LINES = 20

# Helm files take a whitespace-trimming template comment. It renders nothing,
# and trimming keeps rendered manifests and NOTES free of leading blank lines.
# A `#` header instead is emitted into the manifest, and ahead of a leading
# `{{- ... -}}` directive the directive's trim glues the next manifest line onto
# the last comment line, commenting it out — so `check` rejects it.
HELM_OPEN = "{{- /*"
HELM_CLOSE = "*/ -}}"
HELM_OPENERS = frozenset({HELM_OPEN, "{{/*"})
HELM_CLOSERS = frozenset({HELM_CLOSE, "*/}}"})

# A published Markdown file takes the header unprefixed inside an HTML comment
# (the `html` style), each delimiter on its own line. GitHub and MyST render an
# HTML comment as nothing, and a MyST `(label)=` line attaches to the element
# after it, so a header above the label leaves the label on the page's title.
HTML_OPEN = "<!--"
HTML_CLOSE = "-->"

LINE_STYLES = {
    ".rs": "//",
    ".proto": "//",
    ".fbs": "//",
    ".c": "//",
    ".cc": "//",
    ".cpp": "//",
    ".h": "//",
    ".hpp": "//",
    ".lua": "--",
    ".py": "#",
    ".sh": "#",
    ".yaml": "#",
    ".yml": "#",
    ".toml": "#",
    ".ini": "#",
    ".properties": "#",
    ".b64": "#",
    ".proptest-regressions": "#",
    ".dockerignore": "#",
    ".template": "#",
}
REQUIRED_SUFFIXES = frozenset(LINE_STYLES) - {".template"}
REQUIRED_NAMES = frozenset({"Cargo.toml.template"})
IGNORE_FILES = frozenset(
    {".gitignore", ".dockerignore", ".helmignore", ".syncignore", ".publishignore"}
)
# Configuration files known by name that take `#` comments: the ignore files
# and git's attributes file.
NAMED_CONFIG_FILES = IGNORE_FILES | {".gitattributes"}

# Vendored upstream trees: each `<tree>/` sits beside its upstream-file
# manifest `<tree>.SHA256SUMS` (GNU `sha256sum` lines, paths relative to the
# tree, of every file of the published upstream package before the patch) and
# its patch `<tree>.patch` (`vendor_tree_problems`). A re-vendor renames the
# tree here too.
VENDORED_TREES = ("vendor/fred-10.1.0",)

# The classification manifest: the license record of every published file that
# has no comment syntax (`classification_problems`).
CLASSIFICATION = "tools/license-classification.json"
CLASSIFICATION_SCHEMA_VERSION = 1
# Each class, with the optional fields it requires. `sha256` pins the exact
# bytes: the license texts must stay the canonical Apache-2.0 text, and
# captured data was approved for publication as those bytes, so a change needs
# a new approval. `publication` records that approval (role and date, never a
# person's name).
CLASSES = {
    "license-text": ("sha256",),
    "notice": (),
    "lockfile": (),
    "metadata": (),
    "provenance-record": (),
    "synthetic-data": (),
    "captured-data": ("sha256", "publication"),
    "captured-derived": ("publication",),
}
ENTRY_FIELDS = ("class", "owner", "license", "origin", "provenance")
OPTIONAL_ENTRY_FIELDS = ("sha256", "publication")
# An SPDX license expression: license or LicenseRef identifiers joined by
# AND, OR or WITH (no parentheses).
SPDX_EXPRESSION_RE = re.compile(r"[A-Za-z0-9.+-]+(?: (?:AND|OR|WITH) [A-Za-z0-9.+-]+)*")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

# The snapshot builder's own excludes, which apply with or without a
# `.publishignore` (the publish snapshot builder's `TAR_EXCLUDES`).
PUBLISHIGNORE = ".publishignore"
IMPLICIT_PUBLISH_EXCLUDES = ("./.git", "./.publishignore", "./.github-publish", "./github-publish.env")

WALK_SKIP_DIRS = frozenset(
    {
        ".git",
        "target",
        "node_modules",
        "__pycache__",
        ".venv",
        ".venv-linux",
        ".ruff_cache",
        ".pytest_cache",
        ".mypy_cache",
    }
)

# Comment punctuation that may precede a header marker on its line; a comment
# opener there (`/* SPDX-...`) is recognised so that it fails as non-verbatim
# rather than being missed and getting a second header inserted above it.
PREFIX_RE = re.compile(r"\s*(#|//|--|\*|;|/\*+|\{\{-?\s*/\*|--\[=*\[)?\s*")
# A block-comment opener alone on its line, directly above a header.
COMMENT_OPENER_RE = re.compile(r"/\*+|--\[=*\[")
# A Lua long-comment opener (`--[[`, `--[==[`, ...); its level is group 1.
LUA_OPEN_RE = re.compile(r"--\[(=*)\[")
# Comment delimiters stripped before joining a block's text into words.
COMMENT_PUNCTUATION_RE = re.compile(r"//+|/\*+|\*+/|^\s*\*+|#+|--\[=*\[|\]=*\]|--+|\{\{-?|-?\}\}")
LEFTOVER_HINT = (
    " (an upstream file that must keep its own notice takes no NVIDIA header: vendor it in a tree "
    "with an upstream-file manifest, VENDORED_TREES)"
)
# The proprietary header's sentences: left under a header, they are a leftover.
LEGACY_TEXT = tuple(text.strip() for text in LEGACY_BODY[2:] if text.strip())
# Phrases of proprietary notices in general (lower case), not just this
# repository's former wording: under a header, any of them is a leftover.
PROPRIETARY_PHRASES = (
    "licenseref-nvidiaproprietary",
    "retain all intellectual property",
    "strictly prohibited",
    "proprietary rights",
    "express license agreement",
    "confidential",
    "internal use only",
)
# Phrases of an Apache-2.0 license body (lower case). Below the SPDX lines of a
# header whose body does not match, they mean an altered or misaligned body,
# not the short two-line form, so `fix` refuses rather than stacks a new one.
APACHE_PHRASES = (
    "apache license",
    "you may not use this file",
    "www.apache.org/licenses",
    "unless required by applicable law",
    "without warranties or conditions",
    "limitations under the license",
)
# The copyright line wrapped after "&" (the form OSRB rejected); `fix` joins it.
WRAPPED_COPYRIGHT_RE = re.compile(
    r"SPDX-FileCopyrightText: Copyright \(c\) (?P<year>\d{4}(?:-\d{4})?) "
    r"NVIDIA CORPORATION &"
)
WRAPPED_CONTINUATION = "AFFILIATES. All rights reserved."
# Every line either header can have (wrapped copyright continuation included),
# which is what an unprefixed header's comment block consists of.
HEADER_TEXT = frozenset(
    {text.strip() for text in BODY + LEGACY_BODY if text.strip()}
    | {"AFFILIATES. All rights reserved."}
)
CODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=]")
DOCKER_DIRECTIVE_RE = re.compile(r"^#\s*(syntax|escape|check)\s*=", re.IGNORECASE)


class HeaderError(Exception):
    pass


def is_dockerfile(name):
    return name == "Dockerfile" or (
        name.startswith("Dockerfile.") and not name.endswith(".dockerignore")
    )


def is_helm_template(path):
    """A file Helm renders: `.tpl`, manifests and NOTES.txt under `helm/**/templates/`."""
    parts = Path(path).parts
    if "helm" not in parts or "templates" not in parts[parts.index("helm") :]:
        return False
    name = parts[-1]
    return name == "NOTES.txt" or Path(name).suffix in (".yaml", ".yml", ".tpl")


def is_proptest_seed_file(path):
    parts = Path(path).parts
    return "proptest-regressions" in parts[:-1] and Path(path).suffix == ".txt"


# pip requirements files are dependency manifests and accept `#` comments.
REQUIREMENTS_NAME_RE = re.compile(r"(?:^|.*[-_])requirements(?:[-_.][\w.-]*)?\.txt")


def is_requirements_file(path):
    """A pip requirements file: `requirements.txt`, `requirements-dev.txt`,
    `test_requirements.txt`, or any `.txt` in a `requirements/` directory."""
    parts = Path(path).parts
    if Path(path).suffix != ".txt":
        return False
    return bool(REQUIREMENTS_NAME_RE.fullmatch(parts[-1])) or "requirements" in parts[:-1]


def is_markdown(path):
    return Path(path).suffix == ".md"


def requires_header(path):
    """True when the file at `path` (repo-relative, `/`-separated) has comment
    syntax and must carry the header. A pure function of the path: whether the
    file is published, Markdown (`check` adds published Markdown) or an
    upstream file of a vendored tree (`vendor_tree_problems` takes precedence)
    is decided elsewhere."""
    name = Path(path).name
    if (
        name in REQUIRED_NAMES
        or name in NAMED_CONFIG_FILES
        or is_dockerfile(name)
        or is_helm_template(path)
        or is_proptest_seed_file(path)
        or is_requirements_file(path)
    ):
        return True
    return Path(path).suffix in REQUIRED_SUFFIXES


def style_for(path):
    """Comment style for a new header: a line prefix, "helm" or "html"."""
    name = Path(path).name
    if is_helm_template(path):
        return "helm"
    if is_markdown(path):
        return "html"
    if (
        is_dockerfile(name)
        or name in NAMED_CONFIG_FILES
        or is_proptest_seed_file(path)
        or is_requirements_file(path)
    ):
        return "#"
    suffix = Path(path).suffix or name
    if suffix in LINE_STYLES:
        return LINE_STYLES[suffix]
    raise HeaderError(f"{path}: no comment style known for this file type")


def render(prefix, year):
    """Header lines with `prefix` (the text ahead of each line; "" for none)."""
    lines = []
    for text in (COPYRIGHT_TEMPLATE.format(year=year),) + BODY:
        lines.append(prefix + text if text else prefix.rstrip())
    return lines


def new_header(style, year):
    if style == "helm":
        return [HELM_OPEN] + render("", year) + [HELM_CLOSE]
    if style == "html":
        return [HTML_OPEN] + render("", year) + [HTML_CLOSE]
    return render(style + " ", year)


def find_header(lines):
    """Index of the header's copyright line, or None.

    Only a marker preceded by nothing but comment punctuation counts, so prose
    that mentions the SPDX tag (docs, this tool's own tests) is not a header.
    """
    for index, line in enumerate(lines[:SEARCH_LINES]):
        if MARKER in line and PREFIX_RE.fullmatch(line[: line.index(MARKER)]):
            return index
    return None


def preamble_problem(lines, start):
    """What precedes the header at `start` that may not, or None.

    Only a shebang, an encoding declaration, Dockerfile parser directives, blank
    lines and the comment opener the header sits in may come first. Anything
    else (code, another comment, text quoting a header) means the marker found
    is not the file's leading header.
    """
    for index in range(start):
        line = lines[index]
        stripped = line.strip()
        if (
            not stripped
            or (index == 0 and stripped.startswith("#!") and not stripped.startswith("#!["))
            or (index <= 1 and CODING_RE.match(line))
            or DOCKER_DIRECTIVE_RE.match(line)
            or (
                index == start - 1
                and (
                    stripped in HELM_OPENERS
                    or stripped == HTML_OPEN
                    or COMMENT_OPENER_RE.fullmatch(stripped)
                )
            )
        ):
            continue
        return (
            f"line {index + 1}: only a shebang, an encoding line, parser directives or "
            "the header's comment opener may precede the license header"
        )
    return None


def leading_header(lines):
    """Index of the file's leading header (one only allowed lines precede), or
    None. An out-of-scope file is checked only when it has one, so a document
    that quotes the header after some prose is not mistaken for a headed file."""
    start = find_header(lines)
    return None if start is None or preamble_problem(lines, start) else start


def verify(lines, path=None, published=False):
    """Problem description for the header in `lines`, or None when it is verbatim.

    With `path` a Helm template, the header must also be the trimming template
    comment; with `path` a Markdown file and `published`, the HTML comment.
    """
    start = find_header(lines)
    if start is None:
        return "missing license header"
    preamble = preamble_problem(lines, start)
    if preamble:
        return preamble
    line = lines[start]
    prefix = line[: line.index(MARKER)]
    match = COPYRIGHT_RE.fullmatch(line[len(prefix) :])
    if match is None:
        return f"line {start + 1}: copyright line is not verbatim: {line!r}"
    if start + 1 < len(lines) and LEGACY_LICENSE_ID in lines[start + 1]:
        return f"line {start + 2}: NVIDIA proprietary header; this repository is Apache-2.0"
    expected = render(prefix, match.group("year"))[1:]
    actual = lines[start + 1 : start + HEADER_LINES]
    for offset, (want, got) in enumerate(zip(expected, actual)):
        if got.rstrip() != want.rstrip():
            return f"line {start + 2 + offset}: expected {want!r}, found {got!r}"
    if len(actual) < len(expected):
        return "license header is truncated"
    block_end = comment_block_end(lines, start, prefix)
    leftover = leftover_after(lines, start + HEADER_LINES, block_end)
    if leftover is not None:
        index, what = leftover
        return (
            f"line {index + 1}: {what} follows the Apache-2.0 header; remove it by hand"
            f"{LEFTOVER_HINT}"
        )
    if path is not None and is_helm_template(path) and not _wrapped(lines, start, HELM_OPEN, HELM_CLOSE):
        return (
            f"line {start + 1}: a Helm template header must be wrapped in "
            f"`{HELM_OPEN}` ... `{HELM_CLOSE}`, not written as a comment in the output"
        )
    if (
        path is not None
        and published
        and is_markdown(path)
        and not _wrapped(lines, start, HTML_OPEN, HTML_CLOSE)
    ):
        return (
            f"line {start + 1}: a Markdown header must be unprefixed and wrapped in "
            f"`{HTML_OPEN}` ... `{HTML_CLOSE}`, each on its own line, so it renders nothing"
        )
    return None


def comment_block_end(lines, start, prefix):
    """Index just past the comment block that holds the header starting at
    `start`: the run of lines in the header's own comment style, so it never
    reaches code (C `#include`, a Rust `#[...]`, a docstring, a constant).

    - A `//`, `#` or `--` header: lines starting with that marker, blank lines,
      whole `/* ... */` blocks under a `//` or ` * ` header (and `//` lines
      beside a ` * ` header) and whole `--[[ ... ]]` blocks under a `--` header.
    - A header inside a Helm template comment: everything up to the comment's
      closer, then blank lines, further Helm comments and `#` YAML comments
      (which render into the output).
    - A header inside a Markdown HTML comment: everything up to its `-->`,
      then blank lines and further HTML comments.
    - An unprefixed header elsewhere (a document with no comment syntax): its
      own license-text lines and blank lines.

    It bounds `check`'s search for leftovers under a header and `fix`'s check
    for license text under a lone pair of SPDX lines; `fix` never uses it to
    decide which lines to replace.
    """
    marker = prefix.strip()
    opener = lines[start - 1].strip() if start > 0 else ""
    in_helm = marker == "" and opener in HELM_OPENERS
    helm_header = in_helm
    in_html = marker == "" and opener == HTML_OPEN
    html_header = in_html
    # The closer of an open /* */ or --[=[ ]=] block, starting with the one the
    # header itself may sit in.
    block_close = block_closer(opener) if marker in ("", "*") else None
    index = start
    while index < len(lines):
        stripped = lines[index].strip()
        if in_helm:
            in_helm = stripped not in HELM_CLOSERS
        elif in_html:
            in_html = HTML_CLOSE not in stripped
        elif block_close:
            block_close = None if block_close in stripped else block_close
        elif not stripped:
            pass
        elif helm_header and stripped.startswith("{{") and "/*" in stripped:
            # Another template comment; one closed on the same line is complete.
            in_helm = not any(closer in stripped for closer in ("*/}}", "*/ -}}", "*/-}}"))
        elif html_header and stripped.startswith(HTML_OPEN):
            # Another HTML comment; one closed on the same line is complete.
            in_html = HTML_CLOSE not in stripped[len(HTML_OPEN) :]
        elif marker in ("//", "*") and stripped.startswith("/*"):
            block_close = block_closer(stripped)
        elif marker == "--" and LUA_OPEN_RE.match(stripped):
            block_close = block_closer(stripped)
        elif marker and stripped.startswith(marker):
            pass
        elif marker == "#" and stripped == "---":
            pass  # a YAML document marker inside the leading comments
        elif helm_header and stripped.startswith("#"):
            pass  # YAML comments after the template comment render into the output
        elif marker == "*" and stripped.startswith("//"):
            pass  # line comments beside a /* */ header
        elif marker == "" and (MARKER in stripped or stripped in HEADER_TEXT):
            pass
        else:
            break
        index += 1
    return index


def block_closer(line):
    """The closer still owed after `line` if it opens a block comment
    (`/*` or a Lua `--[=*[`) that it does not close, else None."""
    if line.startswith("/*"):
        return None if "*/" in line[2:] else "*/"
    match = LUA_OPEN_RE.match(line)
    if match:
        closer = "]" + match.group(1) + "]"
        return None if closer in line[match.end() :] else closer
    return None


def header_text(lines, index, prefix):
    """`lines[index]` without the header's comment `prefix` (a bare prefix is an
    empty line), or None when it is past the end or not in the header's style."""
    if index >= len(lines):
        return None
    line = lines[index].rstrip()
    if line == prefix.rstrip():
        return ""
    return line[len(prefix) :] if line.startswith(prefix) else None


def nvidia_header_line(line):
    """True when `line` starts an NVIDIA header (its SPDX copyright line)."""
    return (
        MARKER in line
        and "NVIDIA" in line
        and PREFIX_RE.fullmatch(line[: line.index(MARKER)]) is not None
    )


def leftover_after(lines, end, block_end):
    """`(index, what)` of the first line in `lines[end:block_end]` (the rest of
    the header's comment block, after the header) that is NVIDIA proprietary
    text or starts a second NVIDIA header; or None.

    A stacked header (a bad merge, in either order) or a proprietary body left
    under a rewritten identifier would otherwise pass `check`, or survive `fix`.
    Another author's copyright and license lines (code derived from an upstream
    project) and comments that mention the Apache header are left alone.
    """
    words = []  # (word, line index) for the block's comment text, in order
    for index in range(end, block_end):
        line = lines[index]
        lowered = line.lower()
        if any(text in line for text in LEGACY_TEXT) or any(
            phrase in lowered for phrase in PROPRIETARY_PHRASES
        ):
            return index, "NVIDIA proprietary notice"
        if nvidia_header_line(line):
            return index, "second license header"
        words.extend((word, index) for word in COMMENT_PUNCTUATION_RE.sub(" ", lowered).split())
    # A notice wrapped across lines: look for each phrase in the joined text.
    joined = " ".join(word for word, _ in words)
    for phrase in PROPRIETARY_PHRASES:
        at = joined.find(phrase)
        if at >= 0:
            return words[joined[:at].count(" ")][1], "NVIDIA proprietary notice"
    return None


def _wrapped(lines, start, opener, closer):
    """True when the unprefixed header at `start` sits alone between an
    `opener` line and a `closer` line."""
    end = start + HEADER_LINES
    return (
        start > 0
        and lines[start - 1] == opener
        and end < len(lines)
        and lines[end] == closer
        and lines[start].startswith(MARKER)
    )


def header_span(lines, path):
    """`(start, end, prefix, year)` of the existing NVIDIA header; `end` is inclusive."""
    start = find_header(lines)
    line = lines[start]
    prefix = line[: line.index(MARKER)]
    if "NVIDIA" not in line:
        raise HeaderError(
            f"{path}:{start + 1}: header is not an NVIDIA copyright line; fix it by hand"
        )
    # Match the header position by position, so `fix` only ever replaces lines
    # that are literally header text (never comments or code that quote it).
    # Anything it cannot match exactly is refused, never guessed at.
    copyright_text = line[len(prefix) :]
    year = COPYRIGHT_RE.fullmatch(copyright_text)
    ident = start + 1
    if year is None:
        year = WRAPPED_COPYRIGHT_RE.fullmatch(copyright_text)
        if year is None or header_text(lines, ident, prefix) != WRAPPED_CONTINUATION:
            # Another holder on the line, or an unknown wrap: rewriting it
            # would drop text that is not ours to drop.
            raise HeaderError(
                f"{path}:{start + 1}: copyright line is not NVIDIA's (or its known "
                "two-line wrap); fix it by hand"
            )
        ident += 1
    declared = header_text(lines, ident, prefix)
    if declared == LICENSE_ID:
        # A proprietary body under a rewritten identifier is still the old header.
        bodies = (BODY[1:], LEGACY_BODY[1:])
    elif declared == LEGACY_LICENSE_ID:
        # ...and an Apache body under the proprietary identifier is a full header.
        bodies = (LEGACY_BODY[1:], BODY[1:])
    else:
        raise HeaderError(
            f"{path}:{ident + 1}: expected the SPDX-License-Identifier line in the "
            "header's comment style, declaring Apache-2.0 or LicenseRef-NvidiaProprietary; "
            "fix it by hand"
        )
    block_end = comment_block_end(lines, start, prefix)
    last = None
    for body in bodies:
        matched = 0
        while matched < len(body) and header_text(lines, ident + 1 + matched, prefix) == body[matched]:
            matched += 1
        if matched == len(body):
            last = ident + matched
            break
    if last is None:
        # Only the SPDX lines (the older short form) may stand alone. Any license
        # text below them means an altered, truncated or misaligned body, which
        # replacing the SPDX lines alone would leave under the new header.
        rest = range(ident + 1, block_end)
        notice = next((i for i in rest if MARKER in lines[i]), block_end)
        below = [lines[i].lower() for i in range(ident + 1, notice)]
        delimiters = HELM_OPENERS | HELM_CLOSERS | {HTML_OPEN, HTML_CLOSE}
        if declared == LEGACY_LICENSE_ID and any(
            (header_text(lines, i, prefix) or "").strip() not in {"", *delimiters}
            for i in range(ident + 1, block_end)
        ):
            raise HeaderError(
                f"{path}:{ident + 2}: text under a proprietary identifier that is not the "
                "known proprietary body; fix it by hand"
            )
        if any(phrase in text for text in below for phrase in APACHE_PHRASES + PROPRIETARY_PHRASES):
            raise HeaderError(
                f"{path}:{ident + 2}: license header body is altered or truncated; fix it by hand"
            )
        last = ident
    leftover = leftover_after(lines, last + 1, block_end)
    if leftover is not None:
        index, what = leftover
        raise HeaderError(
            f"{path}:{index + 1}: {what} follows the header; fix it by hand{LEFTOVER_HINT}"
        )
    return start, last, prefix, year.group("year")


def insertion_point(lines, path):
    """Index after the lines that must stay first: a shebang (not a Rust `#![...]`
    inner attribute), a Python coding declaration, Dockerfile parser directives.
    Markdown keeps nothing first: the header goes on line 1, above a MyST label."""
    index = 0
    if is_markdown(path):
        return index
    if lines and lines[0].startswith("#!") and not lines[0].startswith("#!["):
        index = 1
    if Path(path).suffix == ".py" and index < len(lines) and CODING_RE.match(lines[index]):
        index += 1
    if is_dockerfile(Path(path).name):
        while index < len(lines) and DOCKER_DIRECTIVE_RE.match(lines[index]):
            index += 1
    return index


def insert(lines, path, year):
    at = insertion_point(lines, path)
    header = new_header(style_for(path), year)
    rest = lines[at:]
    if rest and rest[0].strip():
        header.append("")
    return lines[:at] + header + rest


BOM = "﻿"


def apply(text, path, year, published=False):
    """`text` with a verbatim header, inserting or repairing as needed.

    A repaired header keeps its year and comment style, except in a Helm
    template, and in Markdown with `published`, where any other form is
    replaced by the template comment or the HTML comment (`verify`).
    A leading byte-order mark stays first.
    """
    bom = BOM if text.startswith(BOM) else ""
    return bom + _apply(text[len(bom) :], path, year, published)


def _wrapper(path, published):
    """`(kind, opener, closer, openers, closers)` of the comment a header in
    `path` must be wrapped in (its required delimiters, then every delimiter
    of that comment kind), or None when its own comment prefix is enough."""
    if is_helm_template(path):
        return "Helm", HELM_OPEN, HELM_CLOSE, HELM_OPENERS, HELM_CLOSERS
    if published and is_markdown(path):
        return "Markdown", HTML_OPEN, HTML_CLOSE, frozenset({HTML_OPEN}), frozenset({HTML_CLOSE})
    return None


def _apply(text, path, year, published):
    lines = text.split("\n")
    start = find_header(lines)
    if start is None:
        return "\n".join(insert(lines, path, year))
    preamble = preamble_problem(lines, start)
    if preamble:
        # A header marker below other content: inserting a second header, or
        # rewriting text that merely quotes one, would both be wrong.
        raise HeaderError(f"{path}: {preamble}; fix it by hand")
    if verify(lines, path, published) is None:
        return text
    start, end, prefix, kept_year = header_span(lines, path)
    wrapper = _wrapper(path, published)
    if wrapper and not _wrapped(lines, start, wrapper[1], wrapper[2]):
        kind, _, _, openers, closers = wrapper
        opener = prefix == "" and start > 0 and lines[start - 1].strip() in openers
        closer = end + 1 < len(lines) and lines[end + 1].strip() in closers
        if opener != closer:
            # Removing half a comment block would leak the rest into the output.
            raise HeaderError(
                f"{path}:{start + 1}: {kind} header shares its comment block with other "
                "text; fix it by hand"
            )
        if opener:
            start, end = start - 1, end + 1
        return "\n".join(insert(lines[:start] + lines[end + 1 :], path, kept_year))
    return "\n".join(lines[:start] + render(prefix, kept_year) + lines[end + 1 :])


def tracked_files(root):
    """Repo-relative paths of tracked and non-ignored untracked files.

    Falls back to a directory walk only when git is not installed (the CI image)
    or `root` is not a git checkout; any other git failure (e.g. "dubious
    ownership") is reported, since a walk would also reach ignored files.
    """
    # A missing directory would also raise FileNotFoundError from git, and the
    # walk would then pass on no files at all.
    if not Path(root).is_dir():
        raise HeaderError(f"{root}: no such directory")
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            # Untranslated messages, so "not a git repository" is recognisable.
            env={**os.environ, "LC_ALL": "C"},
        ).stdout
        return sorted({p for p in listed.decode().split("\0") if p})
    except FileNotFoundError:
        pass
    except OSError as error:
        raise HeaderError(f"cannot run git in {root}: {error}") from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode(errors="replace").strip()
        if "not a git repository" not in stderr:
            raise HeaderError(f"git ls-files failed in {root}: {stderr}") from error
    found = []
    for directory, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in WALK_SKIP_DIRS]
        for name in files:
            found.append(Path(directory, name).relative_to(root).as_posix())
    return sorted(found)


def read_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── The published set ───────────────────────────────────────────────────────


def publish_patterns(root):
    """The exclude patterns the publish snapshot builder applies to `root`
    (with `tar --exclude`): its own excludes, then every
    `.publishignore` line that is not blank or a `#` comment, with trailing
    `/`s (and whitespace after them) removed, as the builder cleans them.
    Without a `.publishignore` only the builder's own excludes apply."""
    patterns = list(IMPLICIT_PUBLISH_EXCLUDES)
    path = Path(root) / PUBLISHIGNORE
    if path.is_file():
        for line in path.read_text(encoding="utf-8").split("\n"):
            if re.match(r"\s*(?:#|$)", line):
                continue
            patterns.append(re.sub(r"/+\s*$", "", line))
    return patterns


def is_published(path, patterns):
    """Whether the snapshot builder's `tar --exclude` keeps `path`
    (repo-relative, `/`-separated). A pattern anchored with `./` excludes the
    path whose leading components are its own; an unanchored pattern excludes
    a path holding its components as a contiguous run starting at any
    component boundary. Either way a match on a directory excludes all of it.
    Patterns match literally: the builder's tar would expand `*`, `?` and `[`,
    which this mirror does not, and no `.publishignore` pattern uses them."""
    parts = path.split("/")
    for pattern in patterns:
        if pattern.startswith("./"):
            want = pattern[2:].split("/")
            if parts[: len(want)] == want:
                return False
            continue
        want = pattern.split("/")
        if any(parts[i : i + len(want)] == want for i in range(len(parts) - len(want) + 1)):
            return False
    return True


def published_paths(root, files=None):
    """The subset of `files` (default: `tracked_files(root)`) that the source
    release publishes, mirroring the snapshot builder's `.publishignore`
    handling. With no `.publishignore` (the snapshot itself, or a repository
    that imports it) every file is published. The publish snapshot's own run
    of `check` (verify rule R14) is the backstop for this mirror."""
    files = tracked_files(root) if files is None else files
    patterns = publish_patterns(root)
    return {path for path in files if is_published(path, patterns)}


# ── Vendored upstream trees ─────────────────────────────────────────────────

MANIFEST_LINE_RE = re.compile(r"([0-9a-f]{64})  (\S(?:.*\S)?)")
HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def read_upstream_manifest(path):
    """`[(path, digest)]` of an upstream-file manifest (`<tree>.SHA256SUMS`),
    in file order. Every line must be a GNU `sha256sum` text-mode line,
    `<64 lowercase hex>  <path>`, with a path relative to the tree (no leading
    `./` or `/`, no `..`), each path listed once, and the last line
    terminated. Anything else raises HeaderError: a digest the check cannot
    read cannot vouch for a file."""
    name = Path(path).name
    try:
        text = Path(path).read_bytes().decode("utf-8")
    except FileNotFoundError:
        raise HeaderError(f"{name}: missing: it records which files of the vendored tree are upstream's") from None
    except UnicodeDecodeError:
        raise HeaderError(f"{name}: not UTF-8 text") from None
    if not text.endswith("\n"):
        raise HeaderError(f"{name}: the last line is unterminated")
    entries, seen = [], set()
    for number, line in enumerate(text[:-1].split("\n"), 1):
        match = MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise HeaderError(f"{name}:{number}: not a `<sha256>  <path>` line: {line!r}")
        digest, listed = match.groups()
        if listed.startswith(("./", "/")) or ".." in listed.split("/"):
            raise HeaderError(f"{name}:{number}: path is not relative to the tree: {listed!r}")
        if listed in seen:
            raise HeaderError(f"{name}:{number}: {listed} is listed twice")
        seen.add(listed)
        entries.append((listed, digest))
    return entries


def read_patch_files(path):
    """`(added, edited)`: the paths a vendored tree's patch (`<tree>.patch`,
    unified diffs with `--- a/<path>` / `+++ b/<path>` headers) names, split
    by whether the file's first hunk starts `@@ -0,0` (added) or not (edited).
    Hunks are consumed by their line counts, so a removed line that itself
    starts `-- a/` is never read as a file header. A patch this cannot parse
    raises HeaderError."""
    name = Path(path).name
    try:
        lines = Path(path).read_text(encoding="utf-8").split("\n")
    except FileNotFoundError:
        raise HeaderError(f"{name}: missing: it records which files of the vendored tree NVIDIA adds and edits") from None
    except UnicodeDecodeError:
        raise HeaderError(f"{name}: not UTF-8 text") from None
    if lines[-1] != "":
        raise HeaderError(f"{name}: the last line is unterminated")
    lines.pop()
    added, edited = [], []
    index = 0
    while index < len(lines):
        header = lines[index]
        if not header.startswith("--- a/"):
            raise HeaderError(f"{name}:{index + 1}: expected a `--- a/<path>` file header, found {header!r}")
        target = header[len("--- a/") :]
        if index + 1 >= len(lines) or lines[index + 1] != "+++ b/" + target:
            raise HeaderError(f"{name}:{index + 2}: expected `+++ b/{target}`")
        index += 2
        first = True
        while index < len(lines) and lines[index].startswith("@@ "):
            match = HUNK_RE.match(lines[index])
            if match is None:
                raise HeaderError(f"{name}:{index + 1}: not a hunk header: {lines[index]!r}")
            old = int(match.group(2) or 1)
            new = int(match.group(4) or 1)
            if first:
                (added if lines[index].startswith("@@ -0,0 ") else edited).append(target)
                first = False
            index += 1
            while old or new:
                if index >= len(lines):
                    raise HeaderError(f"{name}: the last hunk of {target} is truncated")
                tag = lines[index][:1]
                if tag not in (" ", "-", "+", "\\"):
                    raise HeaderError(f"{name}:{index + 1}: not a hunk line: {lines[index]!r}")
                if tag in (" ", "-"):
                    old -= 1
                if tag in (" ", "+"):
                    new -= 1
                if old < 0 or new < 0:
                    raise HeaderError(f"{name}:{index + 1}: hunk of {target} is longer than its header says")
                index += 1
            while index < len(lines) and lines[index].startswith("\\"):
                index += 1
        if first:
            raise HeaderError(f"{name}: {target} has a file header but no hunk")
    return added, edited


class VendoredTree:
    """A vendored tree's record: `upstream` maps each file of the published
    upstream package (relative to the tree) to its SHA-256; `added` and
    `edited` are the files the patch adds and edits."""

    def __init__(self, tree, upstream, added, edited):
        self.tree = tree
        self.upstream = upstream
        self.added = frozenset(added)
        self.edited = frozenset(edited)


def load_vendored_tree(root, tree):
    """The record of vendored `tree` (repo-relative) in `root`: its manifest
    `<tree>.SHA256SUMS` and patch `<tree>.patch`. Raises HeaderError when
    either is missing or malformed."""
    upstream = dict(read_upstream_manifest(Path(root) / (tree + ".SHA256SUMS")))
    added, edited = read_patch_files(Path(root) / (tree + ".patch"))
    return VendoredTree(tree, upstream, added, edited)


def vendor_tree_problems(record, full, sub, lines):
    """`(problems, upstream)` for the file `sub` (relative to the tree) of a
    vendored tree, read from `full` (`lines` None when it is not text).

    A file the manifest lists is upstream's: it keeps upstream's notice and
    may not carry NVIDIA's header, and its bytes differ from the listed digest
    only if the patch edits it; `upstream` is then True and no other rule
    applies. Any other file is an NVIDIA addition, which the patch must add;
    `upstream` is then False and the published (or internal) rules apply to
    it as to any file. A manifest file absent from the tree is no problem
    (a repository importing the tree may drop one its git ignores)."""
    tree = record.tree
    if sub not in record.upstream:
        if sub in record.added:
            return [], False
        return [
            f"new file under the vendored tree not recorded in the patch: {tree}.SHA256SUMS does "
            f"not list it, so it is an NVIDIA addition, which {tree}.patch must add (and which "
            "carries the header)"
        ], False
    problems = []
    if lines is not None:
        start = leading_header(lines)
        if start is not None and nvidia_header_line(lines[start]):
            problems.append(
                f"line {start + 1}: upstream file must not be relicensed: {tree}.SHA256SUMS lists "
                "it as the upstream package's, so it keeps upstream's notice and takes no NVIDIA "
                "header"
            )
    if sha256_of(full) != record.upstream[sub] and sub not in record.edited:
        problems.append(
            f"modified upstream file not recorded in the patch: its bytes differ from "
            f"{tree}.SHA256SUMS, and {tree}.patch does not edit it"
        )
    return problems, True


# ── The classification manifest ─────────────────────────────────────────────


def load_classification(root, problems):
    """The `files` entries of the classification manifest in `root`, `{}`
    when there is none. A manifest that is not the documented shape adds one
    problem to `problems` and yields no entries, so every file it would have
    classified is reported too."""
    path = Path(root) / CLASSIFICATION
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        problems.append(f"{CLASSIFICATION}: not valid JSON: {error}")
        return {}
    keys = {"schema_version", "description", "files"}
    if (
        not isinstance(data, dict)
        or set(data) != keys
        or type(data["schema_version"]) is not int
        or data["schema_version"] != CLASSIFICATION_SCHEMA_VERSION
        or not isinstance(data["description"], str)
        or not isinstance(data["files"], dict)
    ):
        problems.append(
            f"{CLASSIFICATION}: expected an object with exactly `schema_version` "
            f"({CLASSIFICATION_SCHEMA_VERSION}), `description` (a string) and `files` (an object "
            "of path → entry)"
        )
        return {}
    return data["files"]


def _entry_field_problems(entry):
    """Problems with an entry's fields alone, whatever file it names."""
    problems = []
    unknown = sorted(set(entry) - set(ENTRY_FIELDS) - set(OPTIONAL_ENTRY_FIELDS))
    if unknown:
        problems.append(f"unknown field(s): {', '.join(unknown)}")
    kind = entry.get("class")
    if isinstance(kind, str) and kind and kind not in CLASSES:
        problems.append(f"unknown class {kind!r}; the classes are: {', '.join(CLASSES)}")
    required = ENTRY_FIELDS + CLASSES.get(kind, ()) if isinstance(kind, str) else ENTRY_FIELDS
    missing = [
        field
        for field in required
        if not (isinstance(entry.get(field), str) and entry[field].strip())
    ]
    if missing:
        problems.append(
            f"missing required field(s): {', '.join(missing)} (each a non-empty string)"
        )
    for field in OPTIONAL_ENTRY_FIELDS:
        if field in entry and field not in missing and not (
            isinstance(entry[field], str) and entry[field].strip()
        ):
            problems.append(f"`{field}` must be a non-empty string")
    license_id = entry.get("license")
    if isinstance(license_id, str) and license_id.strip() and not SPDX_EXPRESSION_RE.fullmatch(license_id):
        problems.append(
            f"license {license_id!r} is not an SPDX expression (identifiers joined by AND, OR or WITH)"
        )
    digest = entry.get("sha256")
    if isinstance(digest, str) and digest.strip() and not SHA256_RE.fullmatch(digest):
        problems.append("`sha256` must be 64 lowercase hex digits")
    return problems


def classification_problems(scope):
    """Every problem with the classification manifest's entries. An entry
    fails closed: on an unknown class or field, a missing required field, a
    file that no longer exists (stale), an internal path (`.publishignore`
    already classifies it), an upstream file of a vendored tree (its manifest
    records it), a file with comment syntax or a Markdown file that is not a
    notice (the header cannot be replaced by an entry), a provenance that is
    not a published file, or bytes that no longer match a `sha256` pin."""
    problems = []
    for path in sorted(scope.entries):
        entry = scope.entries[path]
        where = f"{CLASSIFICATION}: {path}"
        if not isinstance(entry, dict):
            problems.append(f"{where}: the entry is not an object")
            continue
        problems.extend(f"{where}: {problem}" for problem in _entry_field_problems(entry))
        full = scope.root / path
        if path not in scope.file_set or full.is_symlink() or not full.is_file():
            problems.append(f"{where}: stale entry: no such file; remove the entry")
            continue
        kind = entry.get("class")
        located = scope.tree_of(path)
        if path not in scope.published:
            problems.append(
                f"{where}: entry for an internal path, which .publishignore already classifies; "
                "remove the entry"
            )
        elif located is not None and located[1] is not None and located[2] in located[1].upstream:
            problems.append(
                f"{where}: an upstream file of {located[0]}, which {located[0]}.SHA256SUMS records; "
                "remove the entry"
            )
        elif requires_header(path):
            problems.append(
                f"{where}: has comment syntax, so it carries the license header, which an entry "
                "cannot replace; remove the entry"
            )
        elif is_markdown(path) and kind != "notice":
            problems.append(
                f"{where}: a Markdown file carries the license header; only a notice (class "
                "`notice`) is classified instead"
            )
        provenance = entry.get("provenance")
        if isinstance(provenance, str) and provenance.strip() and provenance not in scope.published:
            state = "is not published" if provenance in scope.file_set else "does not exist"
            problems.append(f"{where}: provenance {provenance} {state}; name a published file")
        digest = entry.get("sha256")
        if isinstance(digest, str) and SHA256_RE.fullmatch(digest) and sha256_of(full) != digest:
            if kind == "captured-data":
                hint = "captured data changed — re-approve publication and update the entry"
            elif kind == "license-text":
                hint = "a license text must stay the canonical text the entry pins"
            else:
                hint = "restore the file, or update the entry"
            problems.append(f"{where}: hash drift: {hint}")
    return problems


# ── check and fix ───────────────────────────────────────────────────────────


class Scope:
    """What `check` and `fix` know about the files under `root`: the files
    (`tracked_files`), which of them publish, the vendored trees' records and
    the classification entries. Problems reading a record are kept in
    `problems`; a vendored tree whose record cannot be read maps to None, and
    its files are then left alone (the record's problem fails the check)."""

    def __init__(self, root, files=None):
        self.root = Path(root)
        self.files = tracked_files(self.root) if files is None else files
        self.file_set = set(self.files)
        # `fix_action` matches single paths against the same patterns.
        self.patterns = publish_patterns(self.root)
        self.published = published_paths(self.root, self.files)
        self.problems = []
        self.trees = {}
        for tree in VENDORED_TREES:
            if (self.root / tree).is_dir():
                try:
                    self.trees[tree] = load_vendored_tree(self.root, tree)
                except HeaderError as error:
                    self.trees[tree] = None
                    self.problems.append(f"{tree}: {error}")
        self.entries = load_classification(self.root, self.problems)

    def tree_of(self, path):
        """`(tree, record, path within the tree)` of the vendored tree that
        holds `path`, or None."""
        for tree, record in self.trees.items():
            if path.startswith(tree + "/"):
                return tree, record, path[len(tree) + 1 :]
        return None


def _lines_of(full):
    text = read_text(full)
    return None if text is None else text.removeprefix(BOM).split("\n")


def file_problems(scope, rel):
    """Problems with the file `rel` (repo-relative) under the four outcomes
    the module docstring lists."""
    full = scope.root / rel
    lines = _lines_of(full)
    problems = []
    located = scope.tree_of(rel)
    if located is not None:
        _, record, sub = located
        if record is None:
            return problems
        problems, upstream = vendor_tree_problems(record, full, sub, lines)
        if upstream:
            return problems
    published = rel in scope.published
    classified = rel in scope.entries
    required = requires_header(rel) or (published and is_markdown(rel) and not classified)
    if published and not required and not classified:
        problems.append(
            f"unclassified: it has no comment syntax for the header; add an entry to {CLASSIFICATION}"
        )
    if lines is None:
        if required:
            problems.append("in scope but not UTF-8 text")
        return problems
    if not required and leading_header(lines) is None:
        return problems
    problem = verify(lines, rel, published)
    if problem:
        problems.append(problem)
    return problems


def check(root):
    """Every license problem under `root`, one `path: problem` string each."""
    scope = Scope(root)
    problems = list(scope.problems)
    for rel in scope.files:
        full = scope.root / rel
        if full.is_symlink() or not full.is_file():
            continue
        problems.extend(f"{rel}: {problem}" for problem in file_problems(scope, rel))
    problems.extend(classification_problems(scope))
    return problems


SKIP, HEADER = "skip", "header"


def fix_action(scope, rel, lines, explicit):
    """What `fix` does to `rel` (repo-relative; `lines` None when it is not
    text): HEADER (add or repair the header) or SKIP. A published file with
    no comment syntax and no classification entry is refused when named
    explicitly (`fix PATH`), and skipped by `fix` over the repository, whose
    `check` reports it."""
    located = scope.tree_of(rel)
    if located is not None and (located[1] is None or located[2] in located[1].upstream):
        return SKIP
    published = is_published(rel, scope.patterns)
    classified = rel in scope.entries
    if requires_header(rel) or (published and is_markdown(rel) and not classified):
        return HEADER
    if lines is not None and leading_header(lines) is not None:
        return HEADER
    if explicit and published and not classified and not is_markdown(rel):
        raise HeaderError(f"{rel}: no comment syntax: classify it in {CLASSIFICATION}")
    return SKIP


def _fix(scope, names, year, explicit):
    """Apply `fix_action` to each `(full path, name, inside the root)`."""
    changed = []
    for full, name, inside in names:
        text = read_text(full)
        lines = None if text is None else text.removeprefix(BOM).split("\n")
        if inside:
            action = fix_action(scope, name, lines, explicit)
            published = is_published(name, scope.patterns)
        else:
            # A file outside the repository (a generator's staging copy) is
            # treated as published: it is headed if it has comment syntax or
            # is Markdown, and refused otherwise.
            published = True
            if requires_header(name) or is_markdown(name) or (
                lines is not None and leading_header(lines) is not None
            ):
                action = HEADER
            else:
                raise HeaderError(f"{name}: no comment syntax: classify it in {CLASSIFICATION}")
        if action == SKIP:
            continue
        if text is None:
            raise HeaderError(f"{name}: not UTF-8 text")
        updated = apply(text, name, year, published)
        if updated != text:
            full.write_text(updated, encoding="utf-8")
            changed.append(name)
    return changed


def fix_paths(paths, year, root=None):
    """Header each file in `paths`; names inside `root` are resolved repo-relative.

    Repo-relative naming is what selects the comment style and applies the
    published, vendored-tree and classification rules, so an
    explicit path behaves like the same file reached by `fix_repo`, except
    that a published file with no comment syntax and no classification entry
    is refused.
    """
    root = (root or ROOT).resolve()
    scope = Scope(root)
    names = []
    for path in paths:
        full = Path(path).resolve()
        try:
            names.append((full, full.relative_to(root).as_posix(), True))
        except ValueError:
            names.append((full, str(full), False))
    return _fix(scope, names, year, explicit=True)


def fix_repo(root, year):
    scope = Scope(root)
    names = []
    for rel in scope.files:
        full = scope.root / rel
        if full.is_symlink() or not full.is_file():
            continue
        names.append((full, rel, True))
    return _fix(scope, names, year, explicit=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "check", help="fail if a header is missing or not verbatim, or a published file has no license record"
    )
    fix = sub.add_parser("fix", help="add, repair or relicense headers in place")
    fix.add_argument("--year", default=str(datetime.date.today().year))
    fix.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.command == "check":
            problems = check(ROOT)
            for problem in problems:
                print(problem, file=sys.stderr)
            if problems:
                print(
                    f"{len(problems)} license problem(s) above. Run `python3 tools/license_headers.py fix` "
                    f"to add or repair headers and commit the result; a published file with no comment "
                    f"syntax needs an entry in {CLASSIFICATION} instead. Published Markdown takes the "
                    f"header unprefixed inside an HTML comment (`{HTML_OPEN}` ... `{HTML_CLOSE}`) "
                    f"starting on line 1, which `fix` writes.",
                    file=sys.stderr,
                )
                return 1
            print(
                "Every file that needs the NVIDIA Apache-2.0 header carries it verbatim, and every "
                "published file has its license record."
            )
            return 0
        if args.paths:
            changed = fix_paths(args.paths, args.year)
        else:
            changed = fix_repo(ROOT, args.year)
    except HeaderError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for name in changed:
        print(f"updated {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
