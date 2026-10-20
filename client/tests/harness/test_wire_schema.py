# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class WireSchemaTests(unittest.TestCase):
    def test_exact_package_pin_and_vendored_hashes(self):
        directory = ROOT / "source/live/wire/schema"
        provenance = json.loads((directory / "provenance.json").read_text())
        self.assertEqual(provenance["package"], "omniverse.nucleus-objects.schema.linux-x86_64")
        self.assertEqual(provenance["version"], "111.0-dev+gitlab.23672466.8b7c8579")
        self.assertFalse(provenance["fetch_required"])
        self.assertEqual(set(provenance["files"]), {"OmniObjectDelta.fbs", "OmniObjectDelta_generated.h",
                         "ObjectDeltaBuilder.h", "ObjectMultiPartFormat.h", "LICENSE.txt"})
        for name, expected in provenance["files"].items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected, name)
        self.assertEqual({p.name for p in directory.iterdir()}, set(provenance["files"]) | {"provenance.json", "README.md"})
        self.assertNotIn("@", provenance["repository"])

    def test_vendored_flatbuffers_runtime_matches_the_schema_pin(self):
        vendored = ROOT / "source/live/wire/third_party/flatbuffers"
        record = json.loads((vendored / "provenance.json").read_text())
        pin = json.loads((ROOT / "source/live/wire/schema/provenance.json").read_text())["flatbuffers"]
        self.assertEqual((record["package"], record["version"]), (pin["package"], pin["version"]))
        self.assertEqual((record["upstream_version"], record["license"]), ("1.12.0", "Apache-2.0"))
        self.assertFalse(record["fetch_required"])
        headers = {Path(name).name: digest for name, digest in record["files"].items() if name.startswith("include/")}
        self.assertEqual(headers, pin["headers"])
        for name, digest in record["files"].items():
            self.assertEqual(hashlib.sha256((vendored / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual({p.relative_to(vendored).as_posix() for p in vendored.rglob("*") if p.is_file()},
                         set(record["files"]) | {"provenance.json", "README.md"})
        self.assertIn("FlatBuffers", (ROOT.parent / "NOTICE").read_text())
        self.assertIn("source/live/wire/third_party/flatbuffers", (ROOT / "THIRD_PARTY_NOTICES.md").read_text())


if __name__ == "__main__":
    unittest.main()
