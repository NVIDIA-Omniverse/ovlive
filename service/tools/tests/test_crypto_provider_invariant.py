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

"""Tests for `scripts/check-crypto-provider-invariant.sh`, the check that the
dependency graph carries one rustls crypto provider (`ring`) and never
`aws-lc-rs`, run as a subprocess under `sh`.

The script locates the repository as the parent of its own directory, so each
test copies the real script into the `scripts/` of a throwaway repository in a
temp dir: the bytes under test are the committed script, and the `Cargo.lock`
and `Cargo.toml` it reads are the test's own fixture.

`BannedProviderTests` covers the failure path an `aws-lc-rs` package stanza in
`Cargo.lock` takes: the script exits 1, names the stanza, and explains the
invariant in text that a reader of the published tree can act on.

The script is POSIX sh and needs only sh, grep, awk, sed and coreutils. These
tests are stdlib `unittest`, matching the `supply-chain:python-tests` job. Run
from the repo root with:

    python3 -m unittest tools.tests.test_crypto_provider_invariant
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check-crypto-provider-invariant.sh"

#: Bound on one script run. A healthy run takes well under a second; this only
#: turns a hang into a failure.
RUN_TIMEOUT_SECS = 60

#: A `Cargo.lock` whose only package is the banned provider. The script checks
#: for an aws-lc package stanza before anything else in the lockfile, so no
#: other package is needed to reach that failure.
BANNED_LOCK = 'version = 4\n\n[[package]]\nname = "aws-lc-rs"\nversion = "1.0.0"\n'

#: The failure the banned stanza must produce, as the script prints it.
BANNED_FAILURE = 'Cargo.lock contains a banned aws-lc crypto-provider package: name = "aws-lc-rs"'

#: Internal tracking references the explanation must not cite, as the publish
#: check's rule R11 spells them: a ticket or plan ID (`b.xxx`, `t1.xxx`),
#: numbered `Epic`/`Task` plan steps, and `Bee`/`Subtask` plan items.
TRACKING_REFERENCE = re.compile(
    r"(^|[^A-Za-z0-9_./])(b|t[0-9])\.[a-z0-9]{3}([^A-Za-z0-9_(]|$)"
    r"|(^|[^A-Za-z0-9_])(Epic|Task)[- ][0-9]"
    r"|(^|[^A-Za-z0-9_])(Bee|Subtask)s?([^A-Za-z0-9_]|$)",
    re.M,
)


def _flat(text: str) -> str:
    """`text` with every run of whitespace folded to one space, so a phrase
    the script wraps across `echo` lines reads as one."""
    return " ".join(text.split())


@unittest.skipUnless(
    os.name == "posix" and all(shutil.which(tool) for tool in ("sh", "grep", "awk", "sed")),
    "the script under test is POSIX sh and needs sh, grep, awk and sed",
)
class BannedProviderTests(unittest.TestCase):
    """An `aws-lc-rs` stanza in `Cargo.lock` fails the check, and the failure
    explains what pins the AWS crates to `ring` without pointing at anything
    the published tree does not carry."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name) / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        self.script = self.repo / "scripts" / SCRIPT.name
        shutil.copy2(SCRIPT, self.script)
        (self.repo / "Cargo.lock").write_text(BANNED_LOCK, encoding="utf-8")
        # Present only so the script's existence check passes; the lockfile
        # check fails before the script reads it.
        (self.repo / "Cargo.toml").write_text('[package]\nname = "fixture"\n', encoding="utf-8")

    def run_check(self):
        """Run the copied script from outside the repository (it changes into
        the repository itself); returns `(exit code, stdout + stderr)`."""
        proc = subprocess.run(
            ["sh", str(self.script)],
            cwd=self.repo.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=RUN_TIMEOUT_SECS,
        )
        return proc.returncode, proc.stdout

    def test_banned_stanza_fails_naming_the_stanza(self):
        code, output = self.run_check()

        self.assertEqual(code, 1, output)
        self.assertIn(BANNED_FAILURE, output)
        self.assertIn("ERROR: single-rustls-crypto-provider invariant violated.", output)

    def test_failure_explanation_names_the_aws_crates_and_their_ring_pin(self):
        code, output = self.run_check()

        self.assertEqual(code, 1, output)
        self.assertIn(BANNED_FAILURE, output)
        explanation = _flat(output)
        for phrase in (
            "The AWS crates behind IAM backing-store auth ('aws-config', "
            "'aws-smithy-http-client') default to aws-lc-rs",
            "Cargo.toml pins them to the ring backend",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, explanation, output)

    def test_failure_explanation_cites_no_internal_tracking_reference(self):
        code, output = self.run_check()

        self.assertEqual(code, 1, output)
        self.assertIn(BANNED_FAILURE, output)
        self.assertIsNone(TRACKING_REFERENCE.search(output), output)
        for word in ("Epic ", "Task "):
            with self.subTest(word=word):
                self.assertNotIn(word, output)


if __name__ == "__main__":
    unittest.main()
