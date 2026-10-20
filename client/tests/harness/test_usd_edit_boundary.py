# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Broken controls or redirected SDKs must not become diagnostic evidence."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.live.usd_edit_boundary import observation
from tools.live.test import native_identity
from tools.live.scene_identity import tree_hash


class EditBoundaryEvidenceTests(unittest.TestCase):
    def test_every_recorded_sdk_tree_and_supervisor_source_is_revalidated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            driver = root / "probe"
            driver.write_bytes(b"synthetic executable")
            dependencies = root / "python.so"
            dependencies.write_bytes(b"synthetic Python dependency")
            names = ["tools/live/usd_edit_boundary.py", "tools/live/test.py",
                     "tools/live/test_support.py", "tools/live/process_guard.py", "tools/live/sanitizer.py"]
            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic source")
            trees = []
            for index in range(3):
                directory = root / str(index)
                directory.mkdir()
                (directory / "input").write_bytes(b"original")
                trees.append(dict(path=str(directory), sha256=tree_hash(directory)))
            manifest = dict(schema_version=1, boundary="usd_edit_boundary_probe",
                            artifact_sha256=digest(driver),
                            inputs=[dict(path=name, sha256=digest(root / name)) for name in names],
                            dependencies=[dict(path=str(dependencies), sha256=digest(dependencies))],
                            dependency_trees=trees)
            Path(str(driver) + ".manifest.json").write_text(json.dumps(manifest))
            with patch("tools.live.test.ROOT", root), patch("tools.live.scene_identity.inputs", return_value=[]):
                self.assertEqual(native_identity(driver, "usd_edit_boundary_probe"), manifest)
                for tree in trees:
                    changed = Path(tree["path"]) / "input"
                    changed.write_bytes(b"changed")
                    with self.subTest(tree=tree["path"]), self.assertRaisesRegex(ValueError, "SDK differs"):
                        native_identity(driver, "usd_edit_boundary_probe")
                    changed.write_bytes(b"original")
                (root / "tools/live/process_guard.py").write_text("changed")
                with self.assertRaisesRegex(ValueError, "stale source inputs"):
                    native_identity(driver, "usd_edit_boundary_probe")

    def test_move_control_requires_valid_moved_spec_and_exact_types(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk = Path(temporary)
            library = sdk / "libusd_sdf.so"
            library.write_bytes(b"synthetic library identity")
            identity = dict(schema_version=1, boundary="usd_edit_boundary_diagnostic",
                            usd_version=2511, sdf_library=str(library))
            control = dict(mode="move_control", apply_returned_success=True, data_move_calls=1,
                           old_data_present=False, new_data_present=True,
                           retained_path="/Q", retained_dormant=False)
            def output(value):
                return json.dumps(identity) + "\n" + json.dumps(value) + "\n"
            self.assertEqual(observation("move_control", output(control), sdk)["observed"], control)
            for change in (dict(retained_dormant=True), dict(old_data_present=True),
                           dict(new_data_present=False), dict(data_move_calls=True)):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    observation("move_control", output(dict(control, **change)), sdk)

    def test_loaded_library_must_belong_to_selected_sdk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sdk = root / "sdk"
            sdk.mkdir()
            elsewhere = root / "libusd_sdf.so"
            elsewhere.write_bytes(b"different SDK")
            redirected = sdk / "libusd_sdf.so"
            redirected.symlink_to(elsewhere)
            identity = dict(schema_version=1, boundary="usd_edit_boundary_diagnostic",
                            usd_version=2511, sdf_library=str(redirected))
            observed = dict(mode="delegate", notifications=1, permission_after_callback=False,
                            data_set_calls=1, value_was_written=True)
            with self.assertRaises(ValueError):
                observation("delegate", json.dumps(identity) + "\n" + json.dumps(observed), sdk)


if __name__ == "__main__":
    unittest.main()
