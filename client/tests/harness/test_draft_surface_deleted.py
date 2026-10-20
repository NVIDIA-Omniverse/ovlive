# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain USD authoring is the only authoring surface: no draft or read-view API ships."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / 'source/bindings/python/omni/usd/live'


class DraftSurfaceDeletedTests(unittest.TestCase):
    def test_public_cpp_header_has_no_draft_or_view_symbol(self):
        text = (ROOT / 'include/omni/usd/live/LiveUsd.h').read_text(encoding='utf-8')
        for pattern in (r'\bReadView', r'\bacquireReadView\b', r'\bmaterializeLayer\b', r'\bbeginEdit\b',
                        r'\b(?:class|struct) Edit\b', r'\bEditState\b', r'\bgetFrozenLayer\b'):
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, text))

    def test_python_package_has_no_draft_or_view_name(self):
        for name in ('__init__.py', '_host.py', '__init__.pyi', 'Host.cpp', '_kit_host.py', 'sample.py'):
            text = (PYTHON / name).read_text(encoding='utf-8')
            for pattern in (r'\bReadView\b', r'\bacquire_read_view\b', r'\bbegin_edit\b', r'\b(?:class|struct) Edit\b',
                            r'\bfrozen_layer\b', r'\bmaterialize_layer\b', r'\ballocate_edit\b',
                            r'\bprepare_edit\b', r'\bdef edits\b'):
                with self.subTest(file=name, pattern=pattern):
                    self.assertIsNone(re.search(pattern, text))

    def test_draft_and_view_sources_are_gone(self):
        for name in ('UsdSceneAccess.h', 'UsdSceneAccess.cpp'):
            self.assertFalse((ROOT / 'source/live/scene' / name).exists(), name)


if __name__ == '__main__':
    unittest.main()
