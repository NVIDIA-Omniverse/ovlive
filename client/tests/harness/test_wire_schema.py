# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class WireSchemaTests(unittest.TestCase):
    def test_vendored_flatbuffers_runtime_matches_its_record(self):
        vendored = ROOT / "source/live/wire/third_party/flatbuffers"
        record = json.loads((vendored / "provenance.json").read_text())
        self.assertEqual((record["upstream_version"], record["license"]), ("1.12.0", "Apache-2.0"))
        self.assertFalse(record["fetch_required"])
        for name, digest in record["files"].items():
            self.assertEqual(hashlib.sha256((vendored / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual({p.relative_to(vendored).as_posix() for p in vendored.rglob("*") if p.is_file()},
                         set(record["files"]) | {"provenance.json", "README.md"})
        self.assertIn("FlatBuffers", (ROOT.parent / "NOTICE").read_text())
        self.assertIn("source/live/wire/third_party/flatbuffers", (ROOT / "THIRD_PARTY_NOTICES.md").read_text())


if __name__ == "__main__":
    unittest.main()
