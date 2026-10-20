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
"""Insertion, repair, relicensing, scope and CI-gate regressions for tools/license_headers.py:
the header itself, the published set, Markdown headers, vendored trees and the
classification manifest."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import license_headers as lh


ROOT = Path(__file__).resolve().parents[2]

# NVIDIA's standard Apache-2.0 header, verbatim, `#` style.
APPROVED_HASH = """\
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
# limitations under the License."""

# The NVIDIA proprietary header the repository carried before Apache-2.0.
LEGACY_HASH = """\
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited."""

# SHA-256 of the canonical Apache License 2.0 text as apache.org publishes it
# (https://www.apache.org/licenses/LICENSE-2.0.txt), unmodified — the appendix
# template line `Copyright [yyyy] [name of copyright owner]` stays as is.
# LICENSE.txt and oss/OSS_LICENSE.txt are both that text; NVIDIA's attribution
# is in NOTICE, not in the license text.
APACHE_LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
LICENSE_TEXTS = ("LICENSE.txt", "oss/OSS_LICENSE.txt")

WRAPPED = """\
#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Script description.
set -e
"""


def header(prefix="#", year="2026"):
    return "\n".join(lh.new_header(prefix, year))


def unprefixed(text):
    return re.sub(r"^# ?", "", text, flags=re.M)


def legacy(prefix="#"):
    """The proprietary header in comment style `prefix`, or "helm"."""
    if prefix == "helm":
        return "{{- /*\n" + unprefixed(LEGACY_HASH) + "\n*/ -}}"
    return re.sub(r"^#", prefix, LEGACY_HASH, flags=re.M)


class RenderTests(unittest.TestCase):
    def test_hash_style_is_the_approved_text_verbatim(self):
        self.assertEqual(header("#"), APPROVED_HASH)

    def test_line_styles_differ_only_in_comment_prefix(self):
        for prefix in ("//", "--"):
            rendered = header(prefix)
            self.assertEqual(
                re.sub(rf"^{re.escape(prefix)}", "#", rendered, flags=re.M), APPROVED_HASH
            )

    def test_helm_style_carries_the_same_text_unprefixed(self):
        self.assertEqual(header("helm"), "{{- /*\n" + unprefixed(APPROVED_HASH) + "\n*/ -}}")

    def test_markdown_style_is_the_same_text_unprefixed_in_an_html_comment(self):
        self.assertEqual(header("html"), "<!--\n" + unprefixed(APPROVED_HASH) + "\n-->")


class InsertTests(unittest.TestCase):
    def test_inserts_at_top_with_blank_separator(self):
        self.assertEqual(
            lh.apply("use std::fs;\n", "src/a.rs", "2026"),
            header("//") + "\n\nuse std::fs;\n",
        )

    def test_keeps_shebang_and_coding_lines_first(self):
        text = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\nimport os\n"
        self.assertEqual(
            lh.apply(text, "tools/a.py", "2026"),
            "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"
            + header("#")
            + "\n\nimport os\n",
        )

    def test_keeps_dockerfile_parser_directives_first(self):
        text = "# syntax=docker/dockerfile:1\nFROM scratch\n"
        self.assertEqual(
            lh.apply(text, "Dockerfile", "2026"),
            "# syntax=docker/dockerfile:1\n" + header("#") + "\n\nFROM scratch\n",
        )

    def test_no_double_blank_line_when_content_already_starts_blank(self):
        self.assertEqual(
            lh.apply("\nlocal x = 1\n", "s.lua", "2026"),
            header("--") + "\n\nlocal x = 1\n",
        )

    def test_helm_templates_use_a_trimming_template_comment(self):
        # A `#` header ahead of a leading `{{- ... -}}` is chomped onto the
        # manifest's first line and comments it out; `{{- /* */ -}}` renders nothing.
        text = '{{- $x := required "x" .Values.x -}}\napiVersion: v1\n'
        for path in (
            "helm/chart/templates/job.yaml",
            "helm/chart/templates/_helpers.tpl",
            "helm/chart/templates/NOTES.txt",
        ):
            with self.subTest(path=path):
                self.assertEqual(lh.apply(text, path, "2026"), header("helm") + "\n\n" + text)

    def test_year_is_used_for_new_headers(self):
        self.assertIn("(c) 2031 NVIDIA", lh.apply("x = 1\n", "a.py", "2031"))

    def test_verbatim_header_is_left_untouched(self):
        text = header("#") + "\n\nimport os\n"
        self.assertEqual(lh.apply(text, "a.py", "2031"), text)


class RepairTests(unittest.TestCase):
    def test_wrapped_entity_line_is_joined_and_year_range_kept(self):
        out = lh.apply(WRAPPED, "scripts/a.sh", "2031")
        self.assertEqual(
            out,
            "#!/bin/bash\n"
            + header("#", "2025-2026")
            + "\n#\n# Script description.\nset -e\n",
        )
        self.assertIsNone(lh.verify(out.split("\n")))

    def test_two_line_spdx_gains_the_paragraph_and_keeps_following_text(self):
        text = (
            "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
            "All rights reserved.\n"
            "# SPDX-License-Identifier: LicenseRef-NvidiaProprietary\n"
            "#\n"
            '"""Docstring."""\n'
        )
        self.assertEqual(
            lh.apply(text, "a.py", "2031"), header("#") + '\n#\n"""Docstring."""\n'
        )

    def test_two_line_apache_spdx_gains_the_paragraph(self):
        text = (
            "#!/usr/bin/env bash\n"
            "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
            "All rights reserved.\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
            "#\n"
            "# Guard description.\n"
        )
        self.assertEqual(
            lh.apply(text, "scripts/a.sh", "2031"),
            "#!/usr/bin/env bash\n" + header("#") + "\n#\n# Guard description.\n",
        )

    def test_non_trimming_helm_comment_is_replaced_and_its_text_repaired(self):
        text = (
            "{{/*\nSPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION &\n"
            "AFFILIATES. All rights reserved.\n"
            "SPDX-License-Identifier: LicenseRef-NvidiaProprietary\n*/}}\n"
            '{{- define "x" -}}\n'
        )
        self.assertEqual(
            lh.apply(text, "helm/c/templates/_helpers.tpl", "2031"),
            header("helm", "2025") + '\n\n{{- define "x" -}}\n',
        )

    def test_refuses_a_helm_comment_block_shared_with_other_text(self):
        # Converting would drop the opener but leave the rest of the block and
        # its closer in the rendered manifest.
        text = (
            "{{/*\n" + unprefixed(APPROVED_HASH) + "\n\nService for the chart.\n*/}}\n"
            "apiVersion: v1\n"
        )
        with self.assertRaisesRegex(lh.HeaderError, "shares its comment block"):
            lh.apply(text, "helm/c/templates/service.yaml", "2026")

    def test_hash_header_in_a_helm_template_becomes_a_template_comment(self):
        text = header("#") + '\n\n{{- if .Values.x }}\napiVersion: v1\n{{- end }}\n'
        self.assertEqual(
            lh.apply(text, "helm/c/templates/pdb.yaml", "2031"),
            header("helm") + '\n\n{{- if .Values.x }}\napiVersion: v1\n{{- end }}\n',
        )

    def test_hash_header_outside_helm_templates_is_kept(self):
        text = header("#") + "\n\nkey: value\n"
        self.assertEqual(lh.apply(text, "helm/c/values.yaml", "2031"), text)

    def test_refuses_an_unrecognised_license(self):
        text = (
            "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &\n"
            "# AFFILIATES. All rights reserved.\n"
            "# SPDX-License-Identifier: MIT\n"
        )
        with self.assertRaisesRegex(lh.HeaderError, "declaring Apache-2.0 or LicenseRef-NvidiaProprietary"):
            lh.apply(text, "a.py", "2026")

    def test_refuses_a_third_party_copyright(self):
        text = "// SPDX-FileCopyrightText: 2020 Someone Else\n// SPDX-License-Identifier: MIT\n"
        with self.assertRaisesRegex(lh.HeaderError, "not an NVIDIA copyright"):
            lh.apply(text, "a.rs", "2026")


class RelicenseTests(unittest.TestCase):
    """`fix` replaces the proprietary header with the Apache-2.0 one, in every style."""

    def test_replaces_the_proprietary_header_in_each_line_style(self):
        for prefix, path, body in (
            ("//", "src/a.rs", "use std::fs;\n"),
            ("#", "tools/a.py", '"""Docstring."""\n'),
            ("--", "src/storage/scripts/a.lua", "local x = 1\n"),
        ):
            with self.subTest(path=path):
                out = lh.apply(legacy(prefix) + "\n\n" + body, path, "2031")
                self.assertEqual(out, header(prefix, "2025") + "\n\n" + body)
                self.assertIsNone(lh.verify(out.split("\n"), path))

    def test_replaces_the_proprietary_header_in_a_helm_template(self):
        body = '{{- define "x" -}}\n'
        path = "helm/c/templates/_helpers.tpl"
        out = lh.apply(legacy("helm") + "\n\n" + body, path, "2031")
        self.assertEqual(out, header("helm", "2025") + "\n\n" + body)
        self.assertIsNone(lh.verify(out.split("\n"), path))

    def test_keeps_the_shebang_and_the_text_after_the_header(self):
        text = "#!/bin/bash\n" + legacy("#") + "\n#\n# Script description.\nset -e\n"
        self.assertEqual(
            lh.apply(text, "scripts/a.sh", "2031"),
            "#!/bin/bash\n" + header("#", "2025") + "\n#\n# Script description.\nset -e\n",
        )

    def test_check_names_the_proprietary_header(self):
        lines = (legacy("//") + "\n\nfn main() {}").split("\n")
        self.assertRegex(lh.verify(lines), r"^line 2: NVIDIA proprietary header")


class VerifyTests(unittest.TestCase):
    def test_reports_missing_header(self):
        self.assertEqual(lh.verify(["fn main() {}"]), "missing license header")

    def test_reports_an_altered_word(self):
        lines = header("//").replace("WITHOUT WARRANTIES", "WITHOUT ANY WARRANTIES").split("\n")
        self.assertRegex(lh.verify(lines), r"^line 12: expected")

    def test_reports_a_truncated_header(self):
        self.assertIsNotNone(lh.verify(header("#").split("\n")[:4]))

    def test_reports_mixed_comment_prefixes(self):
        lines = header("//").split("\n")
        lines[3] = "#" + lines[3][2:]
        self.assertRegex(lh.verify(lines), r"^line 4: expected")

    def test_helm_template_rejects_a_comment_that_reaches_the_output(self):
        lines = (header("#") + "\n\n{{- if .Values.x -}}\napiVersion: v1").split("\n")
        self.assertIsNone(lh.verify(lines))
        self.assertRegex(lh.verify(lines, "helm/c/templates/pdb.yaml"), "must be wrapped")
        wrapped = (header("helm") + "\napiVersion: v1").split("\n")
        self.assertIsNone(lh.verify(wrapped, "helm/c/templates/pdb.yaml"))

    def test_prose_mentioning_the_tag_is_not_a_header(self):
        lines = ["Every file starts with an `SPDX-FileCopyrightText` line.", "text"]
        self.assertIsNone(lh.find_header(lines))
        self.assertEqual(lh.verify(lines), "missing license header")


class ScopeTests(unittest.TestCase):
    def test_source_scripts_build_and_config_files_are_in_scope(self):
        for path in (
            "src/main.rs",
            "oodf-ffi/cpp/oodf.cpp",
            "oodf-ffi/include/oodf.h",
            "src/storage/scripts/alloc.lua",
            "tools/a.py",
            "scripts/a.sh",
            "proto/a.proto",
            "helm/c/values.yaml",
            ".gitlab-ci.yml",
            "helm/c/templates/_helpers.tpl",
            "Dockerfile",
            ".dockerignore",
            "conformance_tests/Dockerfile.dockerignore",
            "capabilities-proto/Cargo.toml.template",
            "tests/proptests.proptest-regressions",
            "proptest-regressions/domain/delta.txt",
            "tools/pool.b64",
            "Cargo.toml",
            "rust-toolchain.toml",
            ".cargo/config.toml",
            "helm/c/files/tests/pytest.ini",
            "sonar-project.properties",
            ".gitignore",
            "tools/harness/.gitignore",
            "helm/c/.helmignore",
            ".syncignore",
            ".publishignore",
            ".gitattributes",
            "helm/c/templates/NOTES.txt",
        ):
            with self.subTest(path=path):
                self.assertTrue(lh.requires_header(path))

    def test_documents_data_and_lockfiles_have_no_comment_syntax(self):
        # `requires_header` stays a pure function of the path: published
        # Markdown and classification are decided by `check`, not here.
        for path in (
            "README.md",
            "LICENSE.txt",
            "docs/notes.txt",
            "helm/c/files/data.txt",
            "tools/fixtures/a.json",
            "Cargo.lock",
            "tests/common/.gitkeep",
            "conformance_tests/local-test-token.jwt",
            "tests/fixtures/oodf/a.oodf",
            "docs-internal/validation/run/after.prom",
            ".idea/live_edit.iml",
        ):
            with self.subTest(path=path):
                self.assertFalse(lh.requires_header(path))

    def test_license_texts_are_the_canonical_apache_text_and_the_manifest_pins_it(self):
        # Both copies are the unmodified Apache License 2.0 text, with no header
        # of their own, and their classification entries pin exactly that text.
        for path in LICENSE_TEXTS:
            with self.subTest(path=path):
                license_bytes = (ROOT / path).read_bytes()
                self.assertEqual(hashlib.sha256(license_bytes).hexdigest(), APACHE_LICENSE_SHA256)
                self.assertIsNone(lh.find_header(license_bytes.decode().split("\n")))
        entries = json.loads((ROOT / lh.CLASSIFICATION).read_text())["files"]
        pinned = {
            path: entry.get("sha256")
            for path, entry in entries.items()
            if entry.get("class") == "license-text"
        }
        self.assertEqual(pinned, {path: APACHE_LICENSE_SHA256 for path in LICENSE_TEXTS})

    def test_styles_for_non_suffix_rules(self):
        self.assertEqual(lh.style_for("helm/c/templates/NOTES.txt"), "helm")
        self.assertEqual(lh.style_for("proptest-regressions/domain/delta.txt"), "#")
        self.assertEqual(lh.style_for("helm/c/.helmignore"), "#")
        self.assertEqual(lh.style_for("docs/service/docs/index.md"), "html")

    def test_unknown_file_type_has_no_style(self):
        with self.assertRaisesRegex(lh.HeaderError, "no comment style"):
            lh.style_for("data.xyz")


class VendoredSchemaTests(unittest.TestCase):
    """The OODF schema copy is an ordinary header-required file: the notice its
    upstream carries cannot pass `check` on this path, alone or left under the
    header, and the committed copy verifies."""

    PATH = "oodf-ffi/schema/OmniObjectDelta.fbs"
    # Upstream's leading notice, which oodf-ffi/schema/SOURCE.md records by
    # `**Upstream SHA256:**` and `**Notice lines:**`; the copy replaced it with
    # the header and kept every later line.
    UPSTREAM_NOTICE = (
        "// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.\n"
        "//\n"
        "// NVIDIA CORPORATION and its licensors retain all intellectual property\n"
        "// and proprietary rights in and to this software, related documentation\n"
        "// and any modifications thereto.  Any use, reproduction, disclosure or\n"
        "// distribution of this software and related documentation without an express\n"
        "// license agreement from NVIDIA CORPORATION is strictly prohibited.\n"
        "//\n"
    )
    UPSTREAM_SHA256 = "fe87a05630145c72703412cc8507c371c8edc831f9bc2e4e83605ff482830434"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        committed = (ROOT / self.PATH).read_text()
        # The body: every line after the copy's header (`**Notice lines:**`).
        self.body = "".join(committed.splitlines(keepends=True)[lh.HEADER_LINES :])

    def plant(self, text):
        target = self.root / self.PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target

    def test_upstream_notice_reconstructs_the_pinned_upstream_file(self):
        # The fixture below is upstream's real notice: with the committed body
        # it rebuilds the exact upstream bytes SOURCE.md pins.
        upstream = (self.UPSTREAM_NOTICE + self.body).encode()
        self.assertEqual(hashlib.sha256(upstream).hexdigest(), self.UPSTREAM_SHA256)
        self.assertIn(
            f"**Upstream SHA256:** `{self.UPSTREAM_SHA256}`",
            (ROOT / "oodf-ffi/schema/SOURCE.md").read_text(),
        )

    def test_schema_path_requires_the_header(self):
        self.assertTrue(lh.requires_header(self.PATH))

    def test_upstream_proprietary_notice_on_the_schema_path_fails_check(self):
        self.plant(self.UPSTREAM_NOTICE + self.body)
        self.assertEqual(lh.check(self.root), [f"{self.PATH}: missing license header"])

    def test_header_with_the_upstream_notice_left_below_it_fails_check(self):
        for gap in ("", "\n"):
            with self.subTest(gap=repr(gap)):
                self.plant(header("//", "2019-2026") + "\n" + gap + self.UPSTREAM_NOTICE + self.body)
                problems = lh.check(self.root)
                self.assertEqual(len(problems), 1, problems)
                self.assertRegex(
                    problems[0],
                    rf"^{re.escape(self.PATH)}: line \d+: NVIDIA proprietary notice follows the "
                    "Apache-2.0 header",
                )

    def test_fix_alone_cannot_relicense_the_upstream_notice(self):
        # `fix` inserts the header above upstream's notice, which has no SPDX
        # line to replace; `check` then rejects the notice left below it, so
        # the notice must be removed by hand (SOURCE.md's re-vendor steps).
        self.plant(self.UPSTREAM_NOTICE + self.body)
        self.assertEqual(lh.fix_repo(self.root, "2019-2026"), [self.PATH])
        problems = lh.check(self.root)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("NVIDIA proprietary notice follows the Apache-2.0 header", problems[0])

    def test_committed_schema_verifies(self):
        lines = (ROOT / self.PATH).read_text().split("\n")
        self.assertEqual(lh.leading_header(lines), 0)
        self.assertIsNone(lh.verify(lines, self.PATH, published=True))
        self.assertEqual(lines[: lh.HEADER_LINES], lh.render("// ", "2019-2026"))
        self.plant((ROOT / self.PATH).read_text())
        self.assertEqual(lh.check(self.root), [])


class RepoTests(unittest.TestCase):
    """`check` / `fix` over a directory that is not a git checkout (the CI image has no git).

    Its `.publishignore` makes the two documents internal, where a header is
    optional (`PublishedMarkdownTests` covers published Markdown)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        files = {
            ".publishignore": header("#") + "\n\n./README.md\ndocs/\n",
            "src/ok.rs": header("//") + "\n\nfn a() {}\n",
            "src/missing.rs": "fn b() {}\n",
            "src/legacy.rs": legacy("//") + "\n\nfn c() {}\n",
            "scripts/wrapped.sh": WRAPPED,
            "pytest.ini": WRAPPED.split("\n", 1)[1],
            "README.md": "# Readme\n",
            # Internal, so it needs no header, but a header it carries must
            # still be verbatim.
            "docs/notes.md": legacy("#") + "\n\nNotes.\n",
            # The vendored schema copy is an ordinary header-required file:
            # check reports its proprietary header and fix relicenses it.
            "oodf-ffi/schema/OmniObjectDelta.fbs": legacy("//") + "\ntable T {}\n",
            "target/debug/build.rs": "fn ignored() {}\n",
            # Tool caches git ignores; the walk must skip them too.
            ".ruff_cache/.gitignore": "*\n",
            "conformance_tests/.pytest_cache/.gitignore": "*\n",
        }
        for rel, text in files.items():
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_text(text)
        (self.root / "src/binary.rs").write_bytes(b"\xff\xfe\x00")

    def test_check_reports_each_problem_file(self):
        problems = lh.check(self.root)
        self.assertEqual(
            [p.split(":")[0] for p in problems],
            [
                "docs/notes.md",
                "oodf-ffi/schema/OmniObjectDelta.fbs",
                "pytest.ini",
                "scripts/wrapped.sh",
                "src/binary.rs",
                "src/legacy.rs",
                "src/missing.rs",
            ],
        )

    def test_fix_then_check_is_clean_and_fix_is_idempotent(self):
        (self.root / "src/binary.rs").unlink()
        changed = lh.fix_repo(self.root, "2026")
        self.assertEqual(
            changed,
            [
                "docs/notes.md",
                "oodf-ffi/schema/OmniObjectDelta.fbs",
                "pytest.ini",
                "scripts/wrapped.sh",
                "src/legacy.rs",
                "src/missing.rs",
            ],
        )
        self.assertEqual(lh.check(self.root), [])
        self.assertEqual(
            (self.root / "src/legacy.rs").read_text(), header("//", "2025") + "\n\nfn c() {}\n"
        )
        self.assertEqual(
            (self.root / "docs/notes.md").read_text(), header("#", "2025") + "\n\nNotes.\n"
        )
        self.assertEqual((self.root / ".ruff_cache/.gitignore").read_text(), "*\n")
        self.assertEqual(lh.fix_repo(self.root, "2026"), [])
        self.assertEqual((self.root / "README.md").read_text(), "# Readme\n")
        self.assertEqual(
            (self.root / "oodf-ffi/schema/OmniObjectDelta.fbs").read_text(),
            header("//", "2025") + "\ntable T {}\n",
        )

    def test_fix_refuses_an_in_scope_file_that_is_not_utf8(self):
        # fix_repo must fail loudly, not skip the file and exit 0.
        with self.assertRaisesRegex(lh.HeaderError, "src/binary.rs: not UTF-8 text"):
            lh.fix_repo(self.root, "2026")
        stderr = io.StringIO()
        with patch.object(lh, "ROOT", self.root), contextlib.redirect_stderr(stderr):
            self.assertEqual(lh.main(["fix"]), 1)
        self.assertIn("not UTF-8 text", stderr.getvalue())

    def test_check_command_exits_non_zero_with_guidance(self):
        stderr = io.StringIO()
        with patch.object(lh, "ROOT", self.root), contextlib.redirect_stderr(stderr):
            self.assertEqual(lh.main(["check"]), 1)
        summary = stderr.getvalue().splitlines()[-1]
        self.assertRegex(summary, r"^\d+ license problem\(s\) above\. ")
        for guidance in (
            "tools/license_headers.py fix",
            "needs an entry in tools/license-classification.json",
            "Published Markdown takes the header unprefixed inside an HTML comment (`<!--` ... `-->`)",
            "which `fix` writes",
        ):
            with self.subTest(guidance=guidance):
                self.assertIn(guidance, summary)

    def test_fix_command_on_explicit_paths(self):
        target = self.root / "src/missing.rs"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(lh.main(["fix", "--year", "2026", str(target)]), 0)
        self.assertEqual(target.read_text(), header("//") + "\n\nfn b() {}\n")

    def test_explicit_paths_inside_the_repo_resolve_like_fix_repo(self):
        # An absolute path must still select the Helm style, and relicense the
        # vendored schema copy exactly as `fix_repo` does.
        template = self.root / "helm/c/templates/job.yaml"
        template.parent.mkdir(parents=True)
        body = "{{- $x := .Values.x -}}\napiVersion: v1\n"
        template.write_text(body)
        schema = self.root / "oodf-ffi/schema/OmniObjectDelta.fbs"
        with patch.object(lh, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                lh.main(["fix", "--year", "2026", str(template.resolve()), str(schema.resolve())]),
                0,
            )
        self.assertEqual(template.read_text(), header("helm") + "\n\n" + body)
        self.assertEqual(schema.read_text(), header("//", "2025") + "\ntable T {}\n")


@unittest.skipUnless(shutil.which("git"), "needs git (the CI image has none; the walk covers it)")
class GitListingTests(unittest.TestCase):
    """Untracked files are covered the moment they exist; ignored files are not."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        (self.root / ".gitignore").write_text(header("#") + "\n\nignored.rs\n")
        for name in ("tracked.rs", "new.rs", "ignored.rs"):
            (self.root / name).write_text("fn f() {}\n")
        subprocess.run(["git", "add", "tracked.rs"], cwd=self.root, check=True)

    def test_check_sees_untracked_but_not_ignored_files(self):
        self.assertEqual(
            [p.split(":")[0] for p in lh.check(self.root)], ["new.rs", "tracked.rs"]
        )

    def test_fix_reaches_a_new_unstaged_file(self):
        self.assertEqual(lh.fix_repo(self.root, "2026"), ["new.rs", "tracked.rs"])
        self.assertEqual(lh.check(self.root), [])
        self.assertEqual((self.root / "ignored.rs").read_text(), "fn f() {}\n")


class DamagedHeaderTests(unittest.TestCase):
    """Headers `fix` must replace whole or refuse, never write a new one on top of."""

    def test_rewritten_identifier_over_a_proprietary_body_is_replaced_whole(self):
        text = legacy("#").replace(lh.LEGACY_LICENSE_ID, lh.LICENSE_ID) + "\n\nimport os\n"
        self.assertRegex(lh.verify(text.split("\n")), r"^line 4: expected")
        out = lh.apply(text, "a.py", "2031")
        self.assertEqual(out, header("#", "2025") + "\n\nimport os\n")
        self.assertNotIn("strictly prohibited", out)

    def test_a_proprietary_header_stacked_under_the_apache_one_fails_check(self):
        lines = (header("//") + "\n" + legacy("//") + "\n\nfn a() {}").split("\n")
        # Line 15 is the stacked header's copyright line.
        self.assertRegex(lh.verify(lines), r"^line 15: second license header follows")
        with self.assertRaisesRegex(lh.HeaderError, "second license header follows the header"):
            lh.apply("\n".join(lines), "a.rs", "2026")

    def test_a_proprietary_header_stacked_above_the_apache_one_is_refused(self):
        # The other merge order: converting the first header would leave two.
        text = legacy("#") + "\n\n" + header("#") + "\n\nimport os\n"
        self.assertIsNotNone(lh.verify(text.split("\n")))
        with self.assertRaisesRegex(lh.HeaderError, "second license header follows"):
            lh.apply(text, "a.py", "2026")

    def test_two_apache_headers_fail_check(self):
        lines = (header("#") + "\n\n" + header("#") + "\n\nimport os").split("\n")
        self.assertRegex(lh.verify(lines), r"^line 16: second license header follows")

    def test_a_proprietary_paragraph_left_below_the_header_fails_check(self):
        # The hand-relicensing case: the identifier line was replaced by the
        # Apache header, but the prohibition paragraph was left a few lines below.
        paragraph = "\n".join("# " + text for text in lh.LEGACY_BODY[2:])
        text = header("#") + "\n#\n# Script notes.\n#\n" + paragraph + "\n\nimport os\n"
        self.assertRegex(
            lh.verify(text.split("\n")),
            r"^line 18: NVIDIA proprietary notice follows the Apache-2.0 header",
        )
        with self.assertRaisesRegex(lh.HeaderError, "proprietary notice follows the header"):
            lh.apply(text, "a.py", "2026")

    def test_header_text_quoted_by_code_below_the_header_is_not_a_leftover(self):
        # This module's own shape: a docstring, imports, then constants quoting
        # whole headers. The scan stops at the first line of code.
        text = (
            header("#") + '\n"""Tests."""\n\nimport re\n\nAPPROVED = """\\\n'
            + header("#") + "\n" + legacy("#") + '"""\n'
        )
        self.assertIsNone(lh.verify(text.split("\n")))
        self.assertEqual(lh.apply(text, "tools/tests/test_x.py", "2031"), text)

    def test_proprietary_text_anywhere_in_the_comment_block_is_caught(self):
        # The whole comment block under the header is scanned, however long.
        block = "\n".join(f"# Note {n}." for n in range(20))
        text = header("#") + "\n#\n" + block + "\n# " + lh.LEGACY_LAST_LINE + "\n\nimport os\n"
        self.assertRegex(lh.verify(text.split("\n")), r"^line 36: NVIDIA proprietary notice")

    def test_proprietary_text_after_the_first_code_line_is_out_of_scope(self):
        text = header("#") + "\n\nimport os\n# " + lh.LEGACY_LAST_LINE + "\n"
        self.assertIsNone(lh.verify(text.split("\n")))

    def test_a_proprietary_header_stacked_inside_a_helm_comment_is_caught(self):
        stacked = header("helm").rsplit("\n", 1)[0] + "\n" + unprefixed(LEGACY_HASH) + "\n*/ -}}"
        lines = (stacked + '\n{{- define "x" -}}').split("\n")
        self.assertRegex(lh.verify(lines), r"second license header follows")
        # Inside the Helm comment, prose does not end the scan; only the closer does.
        noted = (
            header("helm").rsplit("\n", 1)[0] + "\n\nHelpers for the chart.\n"
            + lh.LEGACY_LAST_LINE + "\n*/ -}}\n" + '{{- define "x" -}}'
        )
        self.assertRegex(lh.verify(noted.split("\n")), r"NVIDIA proprietary notice follows")

    def test_fix_never_reaches_past_the_comment_block_into_code(self):
        # Header text in code below a short header must not be taken for the
        # header's last line: everything in between would be replaced.
        code = 'fn a() {}\nfn b() {}\nconst S: &str = "limitations under the License.";\nfn c() {}\n'
        short = header("//").split("\n")[:2]
        out = lh.apply("\n".join(short) + "\n\n" + code, "src/a.rs", "2031")
        self.assertEqual(out, header("//") + "\n\n" + code)
        legacy_end = code.replace("limitations under the License.", lh.LEGACY_LAST_LINE)
        out = lh.apply("\n".join(short) + "\n\n" + legacy_end, "src/a.rs", "2031")
        self.assertEqual(out, header("//") + "\n\n" + legacy_end)

    def test_license_text_in_code_does_not_make_a_short_header_truncated(self):
        code = 'const URL: &str = "http://www.apache.org/licenses/LICENSE-2.0";\n'
        short = "\n".join(header("//").split("\n")[:2])
        self.assertEqual(
            lh.apply(short + "\n\n" + code, "src/a.rs", "2031"), header("//") + "\n\n" + code
        )

    def test_a_proprietary_header_stacked_in_a_c_block_comment_is_caught(self):
        block = "/*\n" + re.sub(r"^#", " *", LEGACY_HASH, flags=re.M) + "\n */"
        text = header("//") + "\n\n" + block + "\n#include <x.h>\n"
        self.assertRegex(lh.verify(text.split("\n")), r"^line 17: second license header")
        with self.assertRaisesRegex(lh.HeaderError, "second license header follows"):
            lh.apply(text, "src/a.cpp", "2026")

    def test_preprocessor_and_attribute_lines_are_code_in_slash_comment_files(self):
        for code in (
            '#include <x.h>\n#define MSG "' + lh.LEGACY_LAST_LINE + '"\n',
            '#![allow(dead_code)]\nconst S: &str = "' + lh.LEGACY_LAST_LINE + '";\n',
        ):
            with self.subTest(code=code):
                self.assertIsNone(lh.verify((header("//") + "\n\n" + code).split("\n")))

    def test_another_authors_copyright_line_under_the_header_is_allowed(self):
        # Code derived from an upstream project keeps its author's notice.
        text = (
            header("//") + "\n//\n// SPDX-FileCopyrightText: 2020 Upstream Author\n"
            "// SPDX-License-Identifier: MIT\n\nfn a() {}\n"
        )
        self.assertIsNone(lh.verify(text.split("\n")))

    def test_apache_text_and_upstream_apache_notices_under_the_header_are_allowed(self):
        upstream = (
            "//\n// SPDX-FileCopyrightText: 2020 Upstream Author\n"
            "// SPDX-License-Identifier: Apache-2.0\n"
            + "\n".join("// " + t if t else "//" for t in lh.BODY[2:])
        )
        for below in (
            upstream,
            "//\n//! Writes the `SPDX-License-Identifier: Apache-2.0` header.",
            "//\n// See LICENSE: limitations under the License.",
        ):
            with self.subTest(below=below.splitlines()[1]):
                text = header("//") + "\n" + below + "\n\nfn a() {}\n"
                self.assertIsNone(lh.verify(text.split("\n")))

    def test_fix_on_a_short_header_keeps_every_line_below_it(self):
        # Only the literal header lines are replaced, never comments that quote them.
        short = "\n".join(header("#").split("\n")[:2])
        notes = "\n".join(f"# Note {n}." for n in range(40))
        below = "#\n" + notes + "\n\nimport os\n"
        self.assertEqual(lh.apply(short + "\n" + below, "a.py", "2031"), header("#") + "\n" + below)
        # License wording below lone SPDX lines may be a damaged body: refuse,
        # rather than stack a header on it or guess where it ends.
        quoting = "#\n" + notes + "\n# (see LICENSE: limitations under the License.)\n\nimport os\n"
        with self.assertRaisesRegex(lh.HeaderError, "altered or truncated"):
            lh.apply(short + "\n" + quoting, "a.py", "2031")
        upstream = "//\n// SPDX-FileCopyrightText: 2020 Upstream Author\n// SPDX-License-Identifier: Apache-2.0\n"
        short = "\n".join(header("//").split("\n")[:2])
        self.assertEqual(
            lh.apply(short + "\n" + upstream + "\nfn a() {}\n", "a.rs", "2031"),
            header("//") + "\n" + upstream + "\nfn a() {}\n",
        )

    def test_unprefixed_proprietary_and_wrapped_headers_in_documents_are_repaired(self):
        body = "\nNotes.\n"
        self.assertEqual(
            lh.apply(unprefixed(LEGACY_HASH) + "\n" + body, "docs/x.md", "2031"),
            unprefixed(header("#", "2025")) + "\n" + body,
        )
        verbatim = unprefixed(header("#", "2025-2026"))
        wrapped = verbatim.replace(
            "NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
            "NVIDIA CORPORATION &\nAFFILIATES. All rights reserved.",
            1,
        )
        self.assertEqual(lh.apply(wrapped + "\n" + body, "docs/x.md", "2031"), verbatim + "\n" + body)

    def test_stacked_headers_in_lua_and_c_block_comments_are_caught(self):
        lua = header("--") + "\n--[[\n" + unprefixed(LEGACY_HASH) + "\n]]\nlocal x = 1"
        self.assertRegex(lh.verify(lua.split("\n")), "follows the Apache-2.0 header")
        starred = (
            re.sub(r"^#", " *", APPROVED_HASH, flags=re.M) + "\n */\n/*\n"
            + re.sub(r"^#", " *", LEGACY_HASH, flags=re.M) + "\n */\nint x;"
        )
        self.assertRegex(lh.verify(("/*\n" + starred).split("\n")), "follows the Apache-2.0 header")

    def test_an_unprefixed_header_block_ends_at_the_first_line_of_prose(self):
        # An out-of-scope document whose header is plain text.
        text = unprefixed(APPROVED_HASH) + "\n\nNotes quote: " + lh.LEGACY_LAST_LINE + "\n"
        self.assertIsNone(lh.verify(text.split("\n")))

    def test_a_truncated_header_is_refused_not_stacked(self):
        text = header("//").rsplit("\n", 1)[0] + "\n\nfn a() {}\n"
        self.assertIsNotNone(lh.verify(text.split("\n")))
        with self.assertRaisesRegex(lh.HeaderError, "truncated"):
            lh.apply(text, "a.rs", "2026")


class AmbiguousHeaderTests(unittest.TestCase):
    """What `fix` cannot match exactly it refuses; it never stacks or guesses."""

    def test_a_body_altered_from_its_first_line_is_refused(self):
        for altered in (
            header("#").replace("Licensed under", "Licenced under"),
            header("#").replace('(the "License");', '(the "License")'),
            header("#").replace("Apache-2.0\n#\n", "Apache-2.0\n", 1),
        ):
            with self.subTest(altered=altered.split("\n")[3][:30]):
                with self.assertRaisesRegex(lh.HeaderError, "altered or truncated"):
                    lh.apply(altered + "\n\nimport os\n", "a.py", "2031")

    def test_a_proprietary_identifier_over_an_apache_body_is_replaced_whole(self):
        swapped = header("#", "2025").replace("SPDX-License-Identifier: Apache-2.0", lh.LEGACY_LICENSE_ID)
        self.assertEqual(
            lh.apply(swapped + "\n\nimport os\n", "a.py", "2031"), header("#", "2025") + "\n\nimport os\n"
        )

    def test_unknown_text_under_a_proprietary_identifier_is_refused(self):
        for body in (
            "# This file is confidential and proprietary to NVIDIA.",
            "# Description of the module.",
        ):
            with self.subTest(body=body):
                text = legacy("#").split("\n")[:2] + ["#", body, "", "import os"]
                with self.assertRaisesRegex(lh.HeaderError, "not the known proprietary body"):
                    lh.apply("\n".join(text), "a.py", "2031")

    def test_other_proprietary_wording_under_a_header_fails_check(self):
        for notice in (
            "// NVIDIA CORPORATION and its licensors retain all intellectual property and",
            "// proprietary rights in and to this software; reproduction is strictly prohibited.",
            "// This file is confidential and proprietary to NVIDIA.",
        ):
            with self.subTest(notice=notice):
                text = header("//") + "\n//\n" + notice + "\n\nfn a() {}"
                self.assertRegex(lh.verify(text.split("\n")), "NVIDIA proprietary notice follows")

    def test_a_copyright_line_naming_another_holder_is_refused(self):
        for line in (
            "# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES and "
            "Foo Contributors. All rights reserved.",
            "# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All "
            "rights reserved. Portions Copyright (c) 2019 Foo Inc.",
            "# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.",
        ):
            with self.subTest(line=line[-40:]):
                text = line + "\n" + "\n".join(header("#").split("\n")[1:]) + "\n\nimport os\n"
                with self.assertRaisesRegex(lh.HeaderError, "copyright line is not NVIDIA's"):
                    lh.apply(text, "a.py", "2031")

    def test_an_identifier_out_of_style_names_the_identifier_line(self):
        text = "#  " + header("#")[2:] + "\n\nimport os\n"
        with self.assertRaisesRegex(lh.HeaderError, r"a\.py:2: expected the SPDX-License-Identifier"):
            lh.apply(text, "a.py", "2031")

    def test_proprietary_text_beside_a_header_in_another_comment_style_fails_check(self):
        helm = header("helm") + "\n# " + lh.LEGACY_LAST_LINE + '\n{{- define "x" -}}'
        self.assertRegex(lh.verify(helm.split("\n")), "NVIDIA proprietary notice follows")
        starred = (
            "/*\n" + re.sub(r"^#", " *", APPROVED_HASH, flags=re.M) + "\n */\n"
            "// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual\nint x;"
        )
        self.assertRegex(lh.verify(starred.split("\n")), "NVIDIA proprietary notice follows")

    def test_a_byte_order_mark_stays_first_and_the_header_after_it_is_checked(self):
        bom = lh.BOM
        fixed = lh.apply(bom + legacy("#") + "\n\nimport os\n", "a.py", "2031")
        self.assertEqual(fixed, bom + header("#", "2025") + "\n\nimport os\n")
        verbatim = bom + header("#") + "\n\nimport os\n"
        self.assertEqual(lh.apply(verbatim, "a.py", "2031"), verbatim)
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        (root / "a.py").write_text(bom + legacy("#") + "\n\nimport os\n", encoding="utf-8")
        self.assertRegex(lh.check(root)[0], r"^a\.py: line 2: NVIDIA proprietary header")


class HeaderPlacementAndBlockTests(unittest.TestCase):
    """The header leads the file, and its whole comment block is scanned."""

    def test_anything_but_allowed_preamble_above_the_header_fails_and_is_refused(self):
        for path, text in (
            ("src/a.rs", "// NVIDIA CORPORATION and its licensors retain all intellectual property\n"
             + header("//")),
            ("tools/a.py", "# " + lh.LEGACY_LICENSE_ID + "\n#\n" + header("#")),
            ("scripts/o.sh", "#!/bin/sh\n# Copyright NVIDIA. Proprietary and confidential.\n" + header("#")),
            ("helm/c/templates/m.yaml", "# NVIDIA proprietary\n" + header("helm")),
        ):
            with self.subTest(path=path):
                lines = (text + "\n\nbody\n").split("\n")
                self.assertRegex(lh.verify(lines, path), "may precede the license header")
                with self.assertRaisesRegex(lh.HeaderError, "may precede the license header"):
                    lh.apply(text + "\n\nbody\n", path, "2031")

    def test_header_text_quoted_in_code_near_the_top_is_not_rewritten(self):
        for path, text in (
            ("tests/fixture.rs", 'pub const FIXTURE: &str = r#"\n' + unprefixed(LEGACY_HASH) + '\n"#;\n'),
            ("tools/fixture.py", 'FIXTURE = """\\\n' + unprefixed(LEGACY_HASH) + '\n"""\n'),
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(lh.HeaderError, "may precede the license header"):
                    lh.apply(text, path, "2031")
                self.assertRegex(lh.verify(text.split("\n"), path), "may precede")

    def test_a_document_quoting_the_header_after_prose_is_not_treated_as_headed(self):
        # An internal document (published Markdown needs a header of its own).
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        (root / ".publishignore").write_text(header("#") + "\n\nnotes.md\n")
        (root / "notes.md").write_text("# Notes\n\nThe header reads:\n\n" + legacy("#") + "\n")
        self.assertEqual(lh.check(root), [])
        self.assertEqual(lh.fix_repo(root, "2031"), [])

    def test_a_header_on_a_comment_opener_line_is_recognised_not_doubled(self):
        for path, text in (
            ("oodf-ffi/cpp/a.cpp", "/* " + header("//")[3:].replace("\n// ", "\n * ", 1).split("\n")[0]
             + "\n * SPDX-License-Identifier: Apache-2.0\n */\nint x;\n"),
            ("helm/c/templates/a.yaml", "{{- /* " + unprefixed(header("#")).split("\n")[0]
             + "\nSPDX-License-Identifier: Apache-2.0 */ -}}\napiVersion: v1\n"),
        ):
            with self.subTest(path=path):
                self.assertIsNotNone(lh.find_header(text.split("\n")))
                self.assertIsNotNone(lh.verify(text.split("\n"), path))
                with self.assertRaises(lh.HeaderError):
                    lh.apply(text, path, "2031")

    def test_proprietary_text_inside_every_block_shape_fails_check(self):
        notice = "NVIDIA CORPORATION and its licensors retain all intellectual property"
        starred = re.sub(r"^#", " *", APPROVED_HASH, flags=re.M)
        for name, text in (
            ("unstarred line in the header's C block", "/*\n" + starred + "\n\n   " + notice + "\n */\nint x;"),
            ("unprefixed header in a C block", "/*\n" + unprefixed(APPROVED_HASH) + "\n\n" + notice + "\n*/\nint x;"),
            ("unprefixed header in a Lua block", "--[[\n" + unprefixed(APPROVED_HASH) + "\n" + notice + "\n]]\nlocal x"),
            # A `]]` inside a level-2 block does not close it.
            ("higher-level Lua bracket", header("--") + "\n--[==[\nt[a[1]] = 2\n" + notice + "\n]==]\nlocal x"),
            ("YAML document marker", header("#") + "\n---\n# " + notice + "\nkey: v"),
            ("one-line Helm comment", header("helm") + "\n{{/* " + notice + " */}}\napiVersion: v1"),
            ("NVIDIA CONFIDENTIAL line", header("//") + "\n//\n// NVIDIA CONFIDENTIAL. Internal use only.\nfn a() {}"),
            ("notice wrapped across lines", header("//") + "\n//\n// licensors retain all intellectual\n"
             "// property rights; reproduction without an express\n// license agreement is strictly\n"
             "// prohibited.\nfn a() {}"),
        ):
            with self.subTest(name=name):
                self.assertRegex(lh.verify(text.split("\n")), "NVIDIA proprietary notice follows")

    def test_another_holders_line_does_not_end_the_proprietary_identifier_check(self):
        text = (
            "\n".join(legacy("//").split("\n")[:2])
            + "\n// SPDX-FileCopyrightText: Copyright (c) 2019 Foo Inc.\n"
            "// NVIDIA Confidential - internal use only.\n\nfn a() {}\n"
        )
        with self.assertRaisesRegex(lh.HeaderError, "not the known proprietary body"):
            lh.apply(text, "src/j2.rs", "2031")


class InsertionPointTests(unittest.TestCase):
    def test_rust_inner_attribute_is_not_a_shebang(self):
        text = "#![no_main]\nfn main() {}\n"
        self.assertEqual(lh.apply(text, "fuzz/t.rs", "2026"), header("//") + "\n\n" + text)

    def test_coding_line_is_kept_first_only_in_python(self):
        text = "# coding: values for the chart\nkey: value\n"
        self.assertEqual(lh.apply(text, "helm/c/values.yaml", "2026"), header("#") + "\n\n" + text)

    def test_parser_directives_are_kept_first_only_in_dockerfiles(self):
        text = "# syntax=not-a-directive\nkey = 1\n"
        self.assertEqual(lh.apply(text, "a.toml", "2026"), header("#") + "\n\n" + text)


class HelmCommentTests(unittest.TestCase):
    def test_refuses_a_closer_that_also_ends_other_text(self):
        # The closer belongs to a block that opened before the header with other
        # text; that text also precedes the header, which is refused first.
        text = (
            "{{/* Service for the chart.\n" + unprefixed(header("#", "2025")) + "\n*/ -}}\n"
            "apiVersion: v1\n"
        )
        with self.assertRaisesRegex(lh.HeaderError, "may precede the license header"):
            lh.apply(text, "helm/c/templates/service.yaml", "2026")
        # With a bare opener above, the shared-block guard still applies.
        text = (
            "{{/*\n" + unprefixed(header("#", "2025")) + "\nService for the chart.\n*/ -}}\n"
            "apiVersion: v1\n"
        )
        with self.assertRaisesRegex(lh.HeaderError, "shares its comment block"):
            lh.apply(text, "helm/c/templates/service.yaml", "2026")

    def test_verbatim_header_in_a_non_trimming_comment_is_rewrapped(self):
        body = '{{- define "x" -}}\n'
        text = "{{/*\n" + unprefixed(APPROVED_HASH) + "\n*/}}\n" + body
        path = "helm/c/templates/_helpers.tpl"
        self.assertRegex(lh.verify(text.split("\n"), path), "must be wrapped")
        self.assertEqual(lh.apply(text, path, "2031"), header("helm") + "\n\n" + body)


class RequirementsScopeTests(unittest.TestCase):
    def test_pip_requirements_files_are_in_scope_with_hash_comments(self):
        for path in (
            "requirements.txt",
            "tools/dev-requirements.txt",
            "requirements-dev.txt",
            "test_requirements.txt",
            "requirements/base.txt",
        ):
            with self.subTest(path=path):
                self.assertTrue(lh.requires_header(path))
                self.assertEqual(lh.style_for(path), "#")
        for path in ("docs/valkey-requirements.md", "docs/notes.txt", "tools/prerequisites.txt"):
            with self.subTest(path=path):
                self.assertFalse(lh.requires_header(path))


class TrackedFilesTests(unittest.TestCase):
    """The git branch of `tracked_files`, which CI's git-less image never runs."""

    def run_with(self, result):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        (root / "walked.rs").write_text("fn f() {}\n")
        with patch.object(lh.subprocess, "run", side_effect=[result]) as run:
            listed = lh.tracked_files(root)
        # The exact call: without check=True a failing git would return empty
        # stdout and `check` would pass on no files at all.
        run.assert_called_once()
        self.assertEqual(
            run.call_args.args[0],
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        )
        kwargs = run.call_args.kwargs
        self.assertEqual(
            {key: kwargs[key] for key in ("cwd", "check", "capture_output")},
            {"cwd": root, "check": True, "capture_output": True},
        )
        # The caller's environment (PATH, HOME for safe.directory) with untranslated
        # git messages, so "not a git repository" is recognisable.
        self.assertEqual(kwargs["env"], {**os.environ, "LC_ALL": "C"})
        return listed

    def test_parses_nul_delimited_output_sorted_and_deduplicated(self):
        done = subprocess.CompletedProcess([], 0, stdout=b"b.rs\0dir/a b.rs\0b.rs\0", stderr=b"")
        self.assertEqual(self.run_with(done), ["b.rs", "dir/a b.rs"])

    def test_walks_when_git_is_not_installed(self):
        self.assertEqual(self.run_with(FileNotFoundError("git")), ["walked.rs"])

    def test_walks_outside_a_git_checkout(self):
        error = subprocess.CalledProcessError(
            128, "git", stderr=b"fatal: not a git repository (or any parent)"
        )
        self.assertEqual(self.run_with(error), ["walked.rs"])

    def test_reports_a_missing_root_instead_of_walking_nothing(self):
        with self.assertRaisesRegex(lh.HeaderError, "no such directory"):
            lh.tracked_files(Path(tempfile.gettempdir()) / "license-headers-no-such-dir")

    def test_reports_a_git_that_cannot_run(self):
        with self.assertRaisesRegex(lh.HeaderError, "cannot run git"):
            self.run_with(PermissionError(13, "Permission denied", "git"))

    def test_reports_any_other_git_failure(self):
        error = subprocess.CalledProcessError(
            128, "git", stderr=b"fatal: detected dubious ownership in repository"
        )
        with self.assertRaisesRegex(lh.HeaderError, "dubious ownership"):
            self.run_with(error)


def write_tree(root, files):
    """Write `files` (repo-relative path → str or bytes) under `root`."""
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)


class _TreeCase(unittest.TestCase):
    """A fixture directory per test; it is no git checkout, so the checker walks it."""

    def make_root(self, files):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        write_tree(root, files)
        return root


class PublishedSetTests(_TreeCase):
    """`published_paths` mirrors the snapshot builder's `.publishignore`
    handling (the publish snapshot builder's and tar's exclude matching)."""

    FILES = (
        "README.md",
        "docs-internal/sdd.md",
        "a/b/x.rs",
        "z/a/b/x.rs",
        "b/x.rs",
        "z/b/x.rs",
        "a/c/b/x.rs",
        "foobar/x.rs",
        "x/foo.md",
        "build/ci/v.sh",
        "x/build/ci/v.sh",
        ".github-publish/snapshot/s.rs",
        "github-publish.env",
    )

    def published(self, ignore):
        """The published subset of FILES under a `.publishignore` of `ignore`
        lines (None: no `.publishignore`)."""
        files = {name: "" for name in self.FILES}
        if ignore is not None:
            files[".publishignore"] = "\n".join(ignore) + "\n"
        root = self.make_root(files)
        return lh.published_paths(root, sorted(files))

    def test_excluded_files_and_directories_are_not_published(self):
        # Comments and blank lines are skipped and trailing slashes (and
        # whitespace after them) stripped, as the builder cleans the file.
        published = self.published(["# internal docs", "", "docs-internal/  ", "./README.md"])
        self.assertNotIn("README.md", published)
        self.assertNotIn("docs-internal/sdd.md", published)
        self.assertIn("a/b/x.rs", published)
        # The ignore file itself never publishes.
        self.assertNotIn(".publishignore", published)

    def test_an_anchored_pattern_matches_from_the_root_only(self):
        published = self.published(["./a/b", "./build/ci/"])
        self.assertNotIn("a/b/x.rs", published)
        self.assertNotIn("build/ci/v.sh", published)
        self.assertIn("z/a/b/x.rs", published)
        self.assertIn("x/build/ci/v.sh", published)

    def test_a_bare_pattern_matches_whole_components_at_any_depth(self):
        published = self.published(["b"])
        for path in ("b/x.rs", "z/b/x.rs", "a/b/x.rs", "z/a/b/x.rs", "a/c/b/x.rs"):
            with self.subTest(path=path):
                self.assertNotIn(path, published)
        # A multi-component bare pattern is a contiguous run of components.
        published = self.published(["a/b"])
        self.assertNotIn("a/b/x.rs", published)
        self.assertNotIn("z/a/b/x.rs", published)
        self.assertIn("a/c/b/x.rs", published)
        # Components match whole: no prefix or suffix match.
        published = self.published(["foo"])
        self.assertIn("foobar/x.rs", published)
        self.assertIn("x/foo.md", published)

    def test_without_a_publishignore_every_file_but_the_builders_own_is_published(self):
        published = self.published(None)
        self.assertEqual(
            published, set(self.FILES) - {".github-publish/snapshot/s.rs", "github-publish.env"}
        )

    def test_check_and_fix_take_the_published_set_from_the_mirror(self):
        root = self.make_root({name: "" for name in self.FILES} | {".publishignore": "b\n"})
        with patch.object(lh, "published_paths", wraps=lh.published_paths) as mirror:
            scope = lh.Scope(root)
        mirror.assert_called_once_with(root, scope.files)
        self.assertEqual(scope.published, lh.published_paths(root, scope.files))
        self.assertNotIn("a/b/x.rs", scope.published)

    @unittest.skipUnless(
        (ROOT / ".publishignore").is_file(),
        "a published copy has no .publishignore: every file in it is published",
    )
    def test_the_checkouts_publishignore(self):
        published = lh.published_paths(ROOT)
        for path in (
            "src/main.rs",
            "LICENSE.txt",
            "NOTICE",
            "tools/license_headers.py",
            lh.CLASSIFICATION,
            "vendor/fred-10.1.0/src/lib.rs",
            "docs/service/docs/index.md",
        ):
            with self.subTest(published=path):
                self.assertIn(path, published)
        for path in (
            ".publishignore",
            "CLAUDE.md",
            "IMPLEMENTATION_CHANGELOG.md",
            "docs-internal/sdd.md",
            ".gitlab-ci.yml",
            "oss/PRODUCT_TERMS_OMNIVERSE.txt",
        ):
            with self.subTest(internal=path):
                self.assertNotIn(path, published)


PAGE = "(live-edit-x)=\n# Title\n\nBody.\n"


class PublishedMarkdownTests(_TreeCase):
    """A published Markdown file carries the header in an HTML comment on
    line 1; an internal one needs none."""

    def test_a_published_markdown_file_without_a_header_fails_and_an_internal_one_passes(self):
        root = self.make_root(
            {
                ".publishignore": header("#") + "\n\ninternal/\n",
                "guide.md": PAGE,
                "internal/notes.md": PAGE,
            }
        )
        self.assertEqual(lh.check(root), ["guide.md: missing license header"])

    def test_fix_puts_the_html_header_on_line_one_above_the_label(self):
        out = lh.apply(PAGE, "docs/x.md", "2026", published=True)
        self.assertEqual(out, header("html") + "\n\n" + PAGE)
        self.assertIsNone(lh.verify(out.split("\n"), "docs/x.md", published=True))
        self.assertEqual(lh.apply(out, "docs/x.md", "2031", published=True), out)

    def test_a_header_not_wrapped_in_an_html_comment_fails_and_fix_wraps_it(self):
        for name, text in (
            ("unprefixed", unprefixed(APPROVED_HASH) + "\n\n" + PAGE),
            ("hash-prefixed", header("#") + "\n\n" + PAGE),
        ):
            with self.subTest(name=name):
                lines = text.split("\n")
                self.assertRegex(
                    lh.verify(lines, "docs/x.md", published=True), "must be unprefixed and wrapped"
                )
                self.assertEqual(
                    lh.apply(text, "docs/x.md", "2031", published=True),
                    header("html") + "\n\n" + PAGE,
                )
        root = self.make_root({"guide.md": header("#") + "\n\n" + PAGE})
        self.assertRegex(lh.check(root)[0], r"^guide\.md: line 1: a Markdown header must be")

    def test_an_html_comment_shared_with_other_text_is_refused(self):
        text = "<!--\n" + unprefixed(APPROVED_HASH) + "\nEditor note.\n-->\n" + PAGE
        self.assertIsNotNone(lh.verify(text.split("\n"), "docs/x.md", published=True))
        with self.assertRaisesRegex(lh.HeaderError, "Markdown header shares its comment block"):
            lh.apply(text, "docs/x.md", "2031", published=True)

    def test_proprietary_text_in_the_header_comment_or_the_next_one_fails(self):
        inside = "<!--\n" + unprefixed(APPROVED_HASH) + "\n" + lh.LEGACY_LAST_LINE + "\n-->\n" + PAGE
        after = header("html") + "\n\n<!-- " + lh.LEGACY_LAST_LINE + " -->\n" + PAGE
        for text in (inside, after):
            with self.subTest(text=text.split("\n")[-6]):
                self.assertRegex(
                    lh.verify(text.split("\n"), "docs/x.md", published=True),
                    "NVIDIA proprietary notice follows",
                )

    def test_only_the_comment_opener_may_precede_the_header(self):
        text = "Intro.\n" + header("html") + "\n\n" + PAGE
        self.assertRegex(
            lh.verify(text.split("\n"), "docs/x.md", published=True), "may precede the license header"
        )


TREE = "vendor/fred-10.1.0"
UPSTREAM = {
    "src/lib.rs": b"pub fn a() {}\n",
    "src/edited.rs": b"fn old() {}\n",
    "README.md": b"# fred\n",
    "LICENSE-MIT": b"MIT license text\n",
    # Listed but absent, as in a repository whose git drops it.
    "Cargo.lock": b"# upstream lockfile\n",
}
ADDITION = header("//") + "\n\n//! An NVIDIA addition.\n"


def upstream_manifest(entries):
    return "".join(
        f"{hashlib.sha256(data).hexdigest()}  {path}\n" for path, data in sorted(entries.items())
    )


def edit_hunk(path, old, new):
    return f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-{old}\n+{new}\n"


def add_hunk(path, text):
    lines = text.split("\n")[:-1]
    body = "".join("+" + line + "\n" for line in lines)
    return f"--- a/{path}\n+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n{body}"


class VendoredTreeTests(_TreeCase):
    """A vendored tree's files are upstream's exactly when its manifest lists
    them; the patch records every edit and every NVIDIA addition."""

    def setUp(self):
        files = {f"{TREE}/{path}": data for path, data in UPSTREAM.items() if path != "Cargo.lock"}
        files[f"{TREE}/src/edited.rs"] = b"fn new() {}\n"
        files[f"{TREE}/src/live_edit/mod.rs"] = ADDITION
        files[f"{TREE}.SHA256SUMS"] = upstream_manifest(UPSTREAM)
        files[f"{TREE}.patch"] = edit_hunk("src/edited.rs", "fn old() {}", "fn new() {}") + add_hunk(
            "src/live_edit/mod.rs", ADDITION
        )
        # The two records are classified in the real tree; here they are
        # simply internal.
        files[".publishignore"] = header("#") + f"\n\n./{TREE}.SHA256SUMS\n./{TREE}.patch\n"
        self.root = self.make_root(files)

    def test_the_recorded_tree_passes_with_its_upstream_files_unheaded(self):
        self.assertEqual(lh.check(self.root), [])
        self.assertFalse((self.root / TREE / "Cargo.lock").exists())

    def test_a_new_file_fails_until_headered_and_recorded_in_the_patch(self):
        new = self.root / TREE / "src/live_edit/new.rs"
        new.write_text("fn n() {}\n")
        not_recorded = f"{TREE}/src/live_edit/new.rs: new file under the vendored tree not recorded in the patch"
        problems = lh.check(self.root)
        self.assertEqual(len(problems), 2, problems)
        self.assertTrue(problems[0].startswith(not_recorded), problems)
        self.assertEqual(problems[1], f"{TREE}/src/live_edit/new.rs: missing license header")
        lh.fix_paths([new], "2026", root=self.root)
        problems = lh.check(self.root)
        self.assertEqual(len(problems), 1, problems)
        self.assertTrue(problems[0].startswith(not_recorded), problems)
        with (self.root / f"{TREE}.patch").open("a") as patch_file:
            patch_file.write(add_hunk("src/live_edit/new.rs", new.read_text()))
        self.assertEqual(lh.check(self.root), [])

    def test_a_modified_upstream_file_must_be_one_the_patch_edits(self):
        (self.root / TREE / "src/lib.rs").write_text("pub fn a() { changed() }\n")
        problems = lh.check(self.root)
        self.assertEqual(len(problems), 1, problems)
        self.assertTrue(
            problems[0].startswith(f"{TREE}/src/lib.rs: modified upstream file not recorded in the patch"),
            problems,
        )
        # An upstream file the patch edits may differ from its listed digest.
        (self.root / TREE / "src/edited.rs").write_text("fn newer() {}\n")
        (self.root / TREE / "src/lib.rs").write_bytes(UPSTREAM["src/lib.rs"])
        self.assertEqual(lh.check(self.root), [])

    def test_an_upstream_file_must_not_carry_nvidias_header(self):
        # src/edited.rs is one the patch edits, so only the header is wrong.
        (self.root / TREE / "src/edited.rs").write_text(header("//") + "\n\nfn new() {}\n")
        problems = lh.check(self.root)
        self.assertEqual(len(problems), 1, problems)
        self.assertRegex(problems[0], rf"^{TREE}/src/edited\.rs: line 1: upstream file must not be relicensed")

    def test_a_missing_or_unreadable_record_fails_the_check_once(self):
        manifest = self.root / f"{TREE}.SHA256SUMS"
        patch_path = self.root / f"{TREE}.patch"
        good_manifest, good_patch = manifest.read_text(), patch_path.read_text()
        for name, change, message in (
            ("missing manifest", lambda: manifest.unlink(), "SHA256SUMS: missing"),
            ("missing patch", lambda: patch_path.unlink(), r"\.patch: missing"),
            ("not a sha256sum line", lambda: manifest.write_text("abc README.md\n"), "not a `<sha256>  <path>` line"),
            ("a path listed twice", lambda: manifest.write_text(good_manifest + good_manifest.splitlines(True)[0]), "listed twice"),
            ("unterminated manifest", lambda: manifest.write_text(good_manifest.rstrip("\n")), "unterminated"),
            ("truncated hunk", lambda: patch_path.write_text(good_patch.rsplit("\n", 3)[0] + "\n"), "truncated"),
        ):
            with self.subTest(name=name):
                change()
                problems = lh.check(self.root)
                self.assertEqual(len(problems), 1, problems)
                self.assertRegex(problems[0], rf"^{TREE}: .*{message}")
                manifest.write_text(good_manifest)
                patch_path.write_text(good_patch)
        self.assertEqual(lh.check(self.root), [])

    def test_fix_headers_an_addition_and_leaves_upstream_files_alone(self):
        new = self.root / TREE / "src/live_edit/new.rs"
        new.write_text("fn n() {}\n")
        (self.root / TREE / "src/lib.rs").write_text("pub fn a() { changed() }\n")
        self.assertEqual(lh.fix_repo(self.root, "2026"), [f"{TREE}/src/live_edit/new.rs"])
        self.assertEqual(new.read_text(), header("//") + "\n\nfn n() {}\n")
        self.assertEqual((self.root / TREE / "src/lib.rs").read_text(), "pub fn a() { changed() }\n")
        self.assertEqual(lh.fix_paths([self.root / TREE / "src/lib.rs"], "2026", root=self.root), [])
        self.assertEqual((self.root / TREE / "README.md").read_bytes(), UPSTREAM["README.md"])

    def test_fix_leaves_a_tree_with_an_unreadable_record_alone(self):
        # Without a readable manifest the tree's upstream files cannot be told
        # from NVIDIA's additions, so `fix` headers none of them.
        (self.root / f"{TREE}.SHA256SUMS").write_text("abc README.md\n")
        new = self.root / TREE / "src/live_edit/new.rs"
        new.write_text("fn n() {}\n")
        before = {
            path: path.read_bytes() for path in (self.root / TREE).rglob("*") if path.is_file()
        }
        self.assertEqual(lh.fix_repo(self.root, "2026"), [])
        self.assertEqual(lh.fix_paths([new], "2026", root=self.root), [])
        after = {path: path.read_bytes() for path in (self.root / TREE).rglob("*") if path.is_file()}
        self.assertEqual(after, before)
        problems = lh.check(self.root)
        self.assertEqual(len(problems), 1, problems)
        self.assertRegex(problems[0], rf"^{TREE}: .*not a `<sha256>  <path>` line")

    def test_a_classification_entry_for_an_upstream_file_fails(self):
        # The manifest is the upstream file's record; an entry would be a second one.
        entries = {
            f"{TREE}/LICENSE-MIT": classified("notice", provenance=f"{TREE}/src/live_edit/mod.rs"),
            lh.CLASSIFICATION: classified("metadata", provenance=f"{TREE}/src/live_edit/mod.rs"),
        }
        write_tree(
            self.root,
            {lh.CLASSIFICATION: json.dumps({"schema_version": 1, "description": "x", "files": entries})},
        )
        problems = lh.check(self.root)
        self.assertEqual(len(problems), 1, problems)
        self.assertEqual(
            problems[0],
            f"{lh.CLASSIFICATION}: {TREE}/LICENSE-MIT: an upstream file of {TREE}, which "
            f"{TREE}.SHA256SUMS records; remove the entry",
        )

    def test_a_sibling_of_the_tree_is_ordinary_source(self):
        write_tree(self.root, {f"{TREE}-extra/src/lib.rs": "fn x() {}\n"})
        self.assertEqual(lh.check(self.root), [f"{TREE}-extra/src/lib.rs: missing license header"])


def classified(kind, **fields):
    entry = {
        "class": kind,
        "owner": "NVIDIA CORPORATION & AFFILIATES",
        "license": "Apache-2.0",
        "origin": "A test fixture.",
        "provenance": "src/gen.rs",
    }
    entry.update(fields)
    return entry


CAPTURE = b"\x00\xff captured bytes"


class ClassificationTests(_TreeCase):
    """Every published file with no comment syntax has a valid entry in the
    classification manifest, and every entry fails closed."""

    def setUp(self):
        self.root = self.make_root(
            {
                "src/gen.rs": header("//") + "\n\nfn gen() {}\n",
                "data.json": '{"a": 1}\n',
                "capture.bin": CAPTURE,
                "NOTES.md": "# Notes\n",
                "internal.json": "{}\n",
                ".publishignore": header("#") + "\n\n./internal.json\n",
            }
        )
        self.entries = {
            "data.json": classified("synthetic-data"),
            "capture.bin": classified(
                "captured-data",
                sha256=hashlib.sha256(CAPTURE).hexdigest(),
                publication="Approved for publication by the service owner, 2026-10-15.",
            ),
            "NOTES.md": classified("notice"),
            lh.CLASSIFICATION: classified("metadata"),
        }

    def check_with(self, entries=None, manifest=None):
        if manifest is None:
            manifest = {
                "schema_version": 1,
                "description": "Test manifest.",
                "files": self.entries if entries is None else entries,
            }
        text = manifest if isinstance(manifest, str) else json.dumps(manifest, indent=2) + "\n"
        write_tree(self.root, {lh.CLASSIFICATION: text})
        return lh.check(self.root)

    def assertOneProblem(self, problems, pattern):
        self.assertEqual(len(problems), 1, problems)
        self.assertRegex(problems[0], pattern)

    def test_the_classified_tree_passes(self):
        self.assertEqual(self.check_with(), [])

    def test_a_published_file_with_no_comment_syntax_needs_an_entry(self):
        for path in ("data.json", "capture.bin"):
            with self.subTest(path=path):
                entries = {k: v for k, v in self.entries.items() if k != path}
                self.assertOneProblem(
                    self.check_with(entries),
                    rf"^{re.escape(path)}: unclassified: .* add an entry to tools/license-classification\.json",
                )

    def test_a_stale_or_internal_entry_fails(self):
        for path, pattern in (
            ("gone.json", "stale entry: no such file"),
            ("internal.json", "entry for an internal path"),
        ):
            with self.subTest(path=path):
                self.assertOneProblem(
                    self.check_with({**self.entries, path: classified("synthetic-data")}),
                    rf"^tools/license-classification\.json: {path}: {pattern}",
                )

    def test_an_unknown_class_or_field_or_a_missing_field_fails(self):
        def without(entry, field):
            return {k: v for k, v in entry.items() if k != field}

        for name, entry, pattern in (
            ("unknown class", classified("fixture"), "unknown class 'fixture'"),
            ("unknown field", classified("synthetic-data", sha265="x"), r"unknown field\(s\): sha265"),
            ("missing owner", without(classified("synthetic-data"), "owner"), r"missing required field\(s\): owner"),
            ("empty origin", classified("synthetic-data", origin=" "), r"missing required field\(s\): origin"),
            ("missing class", without(classified("synthetic-data"), "class"), r"missing required field\(s\): class"),
            ("captured without publication", classified("captured-data", sha256="0" * 64), r"missing required field\(s\): publication"),
            ("captured without sha256", classified("captured-data", publication="Approved."), r"missing required field\(s\): sha256"),
            ("license text without sha256", classified("license-text"), r"missing required field\(s\): sha256"),
            ("derived without publication", classified("captured-derived"), r"missing required field\(s\): publication"),
            ("license not SPDX", classified("synthetic-data", license="Apache 2.0"), "license 'Apache 2.0' is not an SPDX expression"),
            ("sha256 not hex", classified("synthetic-data", sha256="ABC"), "`sha256` must be 64 lowercase hex digits"),
            ("not an object", "synthetic-data", "the entry is not an object$"),
        ):
            with self.subTest(name=name):
                problems = self.check_with({**self.entries, "data.json": entry})
                self.assertTrue(problems, name)
                self.assertRegex(problems[0], rf"^tools/license-classification\.json: data\.json: {pattern}")

    def test_hash_drift_fails_with_the_class_specific_fix(self):
        (self.root / "capture.bin").write_bytes(CAPTURE + b"!")
        self.assertOneProblem(
            self.check_with(),
            r"capture\.bin: hash drift: captured data changed — re-approve publication and update the entry$",
        )
        (self.root / "capture.bin").write_bytes(CAPTURE)
        license_text = classified("license-text", sha256=hashlib.sha256(b'{"a": 1}\n').hexdigest())
        self.assertEqual(self.check_with({**self.entries, "data.json": license_text}), [])
        (self.root / "data.json").write_text('{"a": 2}\n')
        self.assertOneProblem(
            self.check_with({**self.entries, "data.json": license_text}),
            r"data\.json: hash drift: a license text must stay the canonical text",
        )

    def test_an_entry_cannot_replace_a_header(self):
        # A file with comment syntax still needs its header, whatever its entry.
        self.assertOneProblem(
            self.check_with({**self.entries, "src/gen.rs": classified("metadata")}),
            r"src/gen\.rs: has comment syntax, so it carries the license header",
        )
        # Markdown is classified only as a notice; any other class needs the header.
        problems = self.check_with({**self.entries, "NOTES.md": classified("synthetic-data")})
        self.assertOneProblem(problems, r"NOTES\.md: a Markdown file carries the license header")
        # Unclassified, the Markdown file needs the header itself.
        entries = {k: v for k, v in self.entries.items() if k != "NOTES.md"}
        self.assertEqual(self.check_with(entries), ["NOTES.md: missing license header"])

    def test_a_provenance_must_be_a_published_file(self):
        for provenance, state in (("nope.rs", "does not exist"), ("internal.json", "is not published")):
            with self.subTest(provenance=provenance):
                entry = classified("synthetic-data", provenance=provenance)
                self.assertOneProblem(
                    self.check_with({**self.entries, "data.json": entry}),
                    rf"data\.json: provenance {re.escape(provenance)} {state}",
                )

    def test_a_malformed_manifest_fails_and_classifies_nothing(self):
        for name, manifest in (
            ("not JSON", "{not json\n"),
            ("wrong version", {"schema_version": 2, "description": "x", "files": {}}),
            ("unknown key", {"schema_version": 1, "description": "x", "files": {}, "extra": 1}),
            ("files not an object", {"schema_version": 1, "description": "x", "files": []}),
        ):
            with self.subTest(name=name):
                problems = self.check_with(manifest=manifest)
                self.assertRegex(problems[0], r"^tools/license-classification\.json: (not valid JSON|expected an object)")
                # With no usable entry every classified file is reported too.
                self.assertTrue(any(p.startswith("data.json: unclassified") for p in problems), problems)


class FixScopeTests(_TreeCase):
    """`fix` headers published Markdown and NVIDIA additions only; it leaves
    internal documents, classified files and upstream files alone, and
    refuses a published file with no comment syntax that has no entry."""

    def setUp(self):
        self.root = self.make_root(
            {
                ".publishignore": header("#") + "\n\ninternal/\n",
                "guide.md": PAGE,
                "internal/notes.md": PAGE,
                "data.json": "{}\n",
                "raw.json": "{}\n",
                # A classified Markdown notice, like ATTRIBUTIONS.md: its entry
                # is its license record, so it takes no header.
                "NOTICE.md": "# Notices\n",
                lh.CLASSIFICATION: json.dumps(
                    {
                        "schema_version": 1,
                        "description": "Test manifest.",
                        "files": {
                            "data.json": classified("synthetic-data", provenance="guide.md"),
                            "NOTICE.md": classified("notice", provenance="guide.md"),
                            lh.CLASSIFICATION: classified("metadata", provenance="guide.md"),
                        },
                    }
                ),
            }
        )

    def main(self, *argv):
        stderr = io.StringIO()
        with patch.object(lh, "ROOT", self.root), contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            code = lh.main(list(argv))
        return code, stderr.getvalue()

    def test_fix_headers_published_markdown_only_and_is_idempotent(self):
        self.assertEqual(lh.fix_repo(self.root, "2026"), ["guide.md"])
        self.assertEqual((self.root / "guide.md").read_text(), header("html") + "\n\n" + PAGE)
        self.assertEqual(lh.fix_repo(self.root, "2026"), [])
        self.assertEqual((self.root / "internal/notes.md").read_text(), PAGE)
        self.assertEqual((self.root / "data.json").read_text(), "{}\n")
        # The unclassified data file is left for `check` to report.
        self.assertEqual((self.root / "raw.json").read_text(), "{}\n")
        self.assertEqual(
            [p.split(":")[0] for p in lh.check(self.root)], ["raw.json"]
        )

    def test_fix_path_refuses_a_published_file_with_no_comment_syntax(self):
        code, stderr = self.main("fix", "--year", "2026", str(self.root / "raw.json"))
        self.assertEqual(code, 1)
        self.assertIn("raw.json: no comment syntax: classify it in tools/license-classification.json", stderr)
        self.assertEqual((self.root / "raw.json").read_text(), "{}\n")

    def test_fix_path_leaves_classified_and_internal_files_alone(self):
        code, _ = self.main(
            "fix", "--year", "2026", str(self.root / "data.json"), str(self.root / "internal/notes.md")
        )
        self.assertEqual(code, 0)
        self.assertEqual((self.root / "data.json").read_text(), "{}\n")
        self.assertEqual((self.root / "internal/notes.md").read_text(), PAGE)

    def test_fix_path_headers_published_markdown(self):
        code, _ = self.main("fix", "--year", "2026", str(self.root / "guide.md"))
        self.assertEqual(code, 0)
        self.assertEqual((self.root / "guide.md").read_text(), header("html") + "\n\n" + PAGE)

    def test_fix_leaves_a_classified_markdown_notice_alone(self):
        self.assertEqual(lh.fix_repo(self.root, "2026"), ["guide.md"])
        self.assertEqual(lh.fix_paths([self.root / "NOTICE.md"], "2026", root=self.root), [])
        self.assertEqual((self.root / "NOTICE.md").read_text(), "# Notices\n")
        self.assertEqual([p.split(":")[0] for p in lh.check(self.root)], ["raw.json"])

    def test_fix_path_outside_the_repo_is_treated_as_published(self):
        # A generator's staging copy: Markdown takes the HTML-comment header,
        # and a file with no comment syntax is refused.
        outside = self.make_root({"staged.md": PAGE, "staged.json": "{}\n"})
        self.assertEqual(
            lh.fix_paths([outside / "staged.md"], "2026", root=self.root), [str((outside / "staged.md").resolve())]
        )
        self.assertEqual((outside / "staged.md").read_text(), header("html") + "\n\n" + PAGE)
        with self.assertRaisesRegex(lh.HeaderError, r"staged\.json: no comment syntax: classify it in"):
            lh.fix_paths([outside / "staged.json"], "2026", root=self.root)
        self.assertEqual((outside / "staged.json").read_text(), "{}\n")


class CheckoutRecordTests(unittest.TestCase):
    """The checkout's own records: the classification manifest, and the
    generator that writes headed Markdown."""

    GENERATOR = ROOT / "examples" / "generate_tail_lineage_goldens.rs"

    def test_the_classification_manifest_has_no_problems(self):
        self.assertEqual(lh.classification_problems(lh.Scope(ROOT)), [])

    def test_attributions_is_the_only_classified_markdown_and_notice_is_a_notice(self):
        entries = json.loads((ROOT / lh.CLASSIFICATION).read_text())["files"]
        self.assertEqual(
            {path: entry["class"] for path, entry in entries.items() if path.endswith(".md")},
            {"ATTRIBUTIONS.md": "notice"},
        )
        self.assertEqual(entries["NOTICE"]["class"], "notice")

    def test_the_fred_patch_entry_names_both_holders(self):
        # The patch's context and removed lines are fred's MIT text; NVIDIA
        # owns only the edits and the added files.
        entry = json.loads((ROOT / lh.CLASSIFICATION).read_text())["files"]["vendor/fred-10.1.0.patch"]
        self.assertIn("Alec Embke", entry["owner"])
        self.assertIn("NVIDIA CORPORATION & AFFILIATES", entry["owner"])
        self.assertEqual(entry["license"], "MIT AND Apache-2.0")

    def test_the_tail_lineage_generator_emits_the_tools_markdown_header(self):
        # Regeneration must reproduce the committed, headed Markdown byte for
        # byte, so the generator's constant is the tool's rendering (year 2026).
        match = re.search(
            r'const MARKDOWN_LICENSE_HEADER: &str = r#"(.*?)"#;',
            self.GENERATOR.read_text(encoding="utf-8"),
            re.S,
        )
        self.assertIsNotNone(match, "no MARKDOWN_LICENSE_HEADER raw-string constant in the generator")
        expected = "\n".join(lh.new_header("html", 2026)) + "\n\n"
        self.assertEqual(match.group(1), expected)
        generated = ["README.md"] + [
            f"{name}.md" for name in ("base", "expected_state", "tail_a", "tail_b1", "tail_b2", "tail_unknown_version")
        ]
        for name in generated:
            with self.subTest(name=name):
                text = (ROOT / "tests/fixtures/tail_lineage" / name).read_text(encoding="utf-8")
                self.assertTrue(text.startswith(expected), name)


if __name__ == "__main__":
    unittest.main()
