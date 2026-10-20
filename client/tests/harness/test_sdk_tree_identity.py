# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selected SDK identities exclude interpreter caches and retain source and binary changes."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.live import build, resolver_identity, scene_identity, wire_identity


class SdkTreeIdentityTests(unittest.TestCase):
    def test_generated_caches_do_not_change_identity_but_sdk_files_do(self):
        for hash_tree in (build._tree_hash, resolver_identity.tree_hash,
                          scene_identity.tree_hash, wire_identity.tree_hash):
            with self.subTest(helper=hash_tree.__module__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                header, library = root / "include/header.h", root / "lib/library.so"
                for path in (header, library):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"selected SDK input")
                original = hash_tree(root)
                for name in ("lib/__pycache__/module.cpython-312.pyc",
                             "lib/nested/__pycache__/ignored.txt", "lib/module.pyc", "lib/module.pyo"):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"generated cache")
                    self.assertEqual(hash_tree(root), original, name)
                    path.write_bytes(b"rewritten generated cache")
                    self.assertEqual(hash_tree(root), original, name)
                for path in (header, library):
                    path.write_bytes(b"changed SDK input")
                    self.assertNotEqual(hash_tree(root), original, path.name)
                    path.unlink()
                    self.assertNotEqual(hash_tree(root), original, path.name)
                    path.write_bytes(b"selected SDK input")
                    self.assertEqual(hash_tree(root), original)
                for name in ("lib/module.py", "include/package.pyc/header.h"):
                    source = root / name
                    source.parent.mkdir(parents=True, exist_ok=True)
                    source.write_bytes(b"SDK source")
                    self.assertNotEqual(hash_tree(root), original, name)
                    source.unlink()
                    self.assertEqual(hash_tree(root), original)

    def test_missing_root_keeps_each_callers_behavior(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            self.assertEqual(scene_identity.tree_hash(missing), hashlib.sha256().hexdigest())
            with self.assertRaisesRegex(ValueError, "resolver SDK directory is missing"):
                resolver_identity.tree_hash(missing)
            with self.assertRaisesRegex(ValueError, "missing SDK tree"):
                wire_identity.tree_hash(missing)
            with self.assertRaises(FileNotFoundError):
                build._tree_hash(missing)

    def test_shared_helper_is_a_source_input_for_consumers(self):
        root = Path(__file__).resolve().parents[2]
        helper = "tools/live/sdk_tree_identity.py"
        for inputs in (scene_identity.inputs, resolver_identity.inputs, wire_identity.inputs):
            with self.subTest(consumer=inputs.__module__):
                self.assertIn(helper, inputs(root))
        for name in ("LiveSceneCore.cmake", "LiveResolverProbe.cmake", "LiveWire.cmake"):
            with self.subTest(cmake=name):
                self.assertIn(helper, (root / "cmake" / name).read_text())

    def test_direct_identity_scripts_import_helper_outside_checkout(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("scene_identity.py", "resolver_identity.py", "wire_identity.py"):
                with self.subTest(script=name):
                    result = subprocess.run([sys.executable, "-I", "-B", str(root / "tools/live" / name), "--help"],
                                            cwd=temporary, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)
