# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Design contract hygiene for the direct-authoring model."""
import pathlib, re, subprocess, unittest

DOCS = pathlib.Path(__file__).resolve().parents[2] / "docs"
DESIGN = DOCS / "ovlive-design.md"
COMPANION_DOCS = ["ovlive-design.md", "ovlive-python.md", "ovlive-kit.md", "ovlive-reference.md"]
FORBIDDEN_PHRASES = [" now ", "no longer", "previously", "currently", "this increment", "checkpoint (", "RECOVERY_REQUIRED", "recovery required", "recovery-required", "recovery gate", "recovery bundle", "recovery archive", "editor record", "ReadView", "beginEdit", "UsdEdit", "SdfEditDraft", "HostMutationFault", "quarantine", "read view", "edit draft", "complete bundle", "recovery manifest", "recovery export", "ExportLayerRecovery", "export_layer_recovery", "live_recovery_inspect", "recovery-usd"]
# The design states contracts; qualification status, milestones and verification records live elsewhere.
STATUS_PATTERN = re.compile(r"\bcarried\b|\(partial\b|\bpartial M\d|\brulings?\b|\bM[0-4]\.[0-9]|verification record|"
                            r"remains? (?:open|unfinished|unqualified|incomplete)|not yet (?:implemented|qualified|integrated)|"
                            r"evidence (?:is|are) recorded|recorded separately in|\bPhase [0-9R]\b", re.IGNORECASE)
TABLE_MILESTONE = re.compile(r"\bM[0-4]\b")
REQUIRED_PHRASES = ["host edit buffer", "flush job", "captured intent", "verified candidate", "SetPermissionToEdit(true)", "stock `SdfData` semantics", "superseded", "FAILED"]


def prose_lines(text):
    """Yield (line number, line) outside fenced code blocks."""
    fenced = False
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced:
            yield number, line


def phrase_pattern(phrase):
    # A lowercase phrase matches only at a word start ("read view" but not "spread view"); it may end
    # inside a longer word ("recovery gate" matches "recovery gateway") unless it is written with a
    # trailing space, which also requires a word end ("now" but not "nowhere"). Identifiers such as
    # "ReadView" match anywhere, including inside "UsdReadView".
    core = phrase.strip()
    left = r"(?<![a-z])" if core[0].islower() else ""
    right = r"(?![a-z])" if phrase.endswith(" ") else ""
    return re.compile(left + re.escape(core) + right, re.IGNORECASE)


PATTERNS = [(phrase, phrase_pattern(phrase)) for phrase in FORBIDDEN_PHRASES]


def forbidden_hits(text):
    return [(number, phrase) for number, line in prose_lines(text)
            for phrase, pattern in PATTERNS if pattern.search(line)]


class DesignContractVocabulary(unittest.TestCase):
    def test_direct_authoring_vocabulary_present(self):
        text = DESIGN.read_text(encoding="utf-8")
        missing = [p for p in REQUIRED_PHRASES if p not in text]
        self.assertEqual(missing, [], f"missing contract vocabulary: {missing}")

    def test_design_states_contracts_not_status(self):
        text = DESIGN.read_text(encoding="utf-8")
        hits = [number for number, line in prose_lines(text) if STATUS_PATTERN.search(line)
                or (line.startswith("|") and TABLE_MILESTONE.search(line))]
        self.assertEqual(hits, [], f"status vocabulary in the design (lines): {hits}")


class CompanionDocVocabulary(unittest.TestCase):
    def test_design_python_and_kit_docs(self):
        for name in COMPANION_DOCS:
            with self.subTest(doc=name):
                hits = forbidden_hits((DOCS / name).read_text(encoding="utf-8"))
                self.assertEqual(hits, [], f"forbidden phrases in {name} (line, phrase): {hits}")

    def test_no_doc_names_a_recovery_surface(self):
        pattern = re.compile(r"omniUsdLiveExportLayerRecovery|export_layer_recovery|live_recovery_inspect|"
                             r"RecoveryCheck|RecoveryRetirement|checkRecovery|recovery-usd|RECOVERY_REQUIRED|"
                             r"recovery_schema_version|recovery_exported|\.live-recovery")
        # The client's guides and the repository-wide documents under the repository root's docs.
        names = subprocess.check_output(["git", "ls-files", "docs/*.md"], cwd=DOCS.parent, text=True).split()
        repository = DOCS.parent.parent
        shared = subprocess.check_output(["git", "ls-files", "docs/*.md"], cwd=repository, text=True).split()
        for path in [*(DOCS.parent / name for name in names), *(repository / name for name in shared)]:
            with self.subTest(doc=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertEqual([number for number, line in prose_lines(text) if pattern.search(line)], [])


if __name__ == "__main__":
    unittest.main()
