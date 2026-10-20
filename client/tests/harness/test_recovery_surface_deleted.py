# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recovery is SdfLayer::Export and reopen: no plugin-specific recovery surface ships."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / 'source/bindings/python/omni/usd/live'


def absent(test, path, patterns):
    text = path.read_text(encoding='utf-8')
    for pattern in patterns:
        with test.subTest(file=str(path.relative_to(ROOT)), pattern=pattern):
            test.assertIsNone(re.search(pattern, text))


class RecoverySurfaceDeletedTests(unittest.TestCase):
    def test_python_package_has_no_recovery_name(self):
        for name in ('__init__.py', '__init__.pyi', '_host.py', '_kit_root.py', '_kit_host.py', 'sample.py',
                     'Host.cpp', 'Native.cpp'):
            absent(self, PYTHON / name, (r'\bexport_layer_recovery\b', r'RecoveryCheck', r'RecoveryRetirement',
                                         r'\bcheck_recovery\b', r'recovery_retirement', r'\bretire_for_recovery\b',
                                         r'checkRecovery', r'retirementResult', r'(?i)recovery retirement',
                                         r'\bcheck_preservation\b', r'\bforget_preservation\b', r'\bpreservations\b',
                                         r'\bdef preserve\b', r'_preserv', r'\bRootPreservation\b', r'\bRecoveryReceipt\b',
                                         r'live_recovery_inspect'))

    def test_public_cpp_header_has_no_recovery_symbol(self):
        absent(self, ROOT / 'include/omni/usd/live/LiveUsd.h',
               (r'RecoveryCheck', r'RecoveryRetirement\b', r'checkRecovery', r'retirementResult',
                r'\bbeginRecoveryRetirement\b', r'\bgetRecoveryRetirement\b', r'(?i)recovery retirement'))

    def test_c_abi_has_no_recovery_export(self):
        absent(self, ROOT / 'include/omni/usd/live/Live.h',
               (r'ExportLayerRecovery', r'RECOVERY_SCHEMA_VERSION', r'\brecovery_schema_version\b'))
        for name in ('source/live/runtime/Live.exports', 'source/live/scene/plugin/LivePlugin.exports'):
            text = (ROOT / name).read_text(encoding='utf-8')
            self.assertNotIn('ExportLayerRecovery', text, name)
            self.assertEqual(len(re.findall(r'omniUsdLive\w+;', text)), 40, name)
        absent(self, PYTHON / '__init__.py', (r'\brecovery_schema_version\b',))

    def test_retired_values_have_no_names(self):
        absent(self, ROOT / 'include/omni/usd/live/Live.h',
               (r'OMNI_USD_LIVE_RECOVERY_REQUIRED\b', r'STATE_RECOVERY_REQUIRED', r'OPERATION_RECOVERY_EXPORT',
                r'RECORD_RECOVERY_EXPORTED'))
        for name in ('__init__.py', '__init__.pyi'):
            absent(self, PYTHON / name, (r'\bRECOVERY_REQUIRED\b',))

    def test_capture_sources_are_gone(self):
        for name in ('LiveSdfRecovery.h', 'LiveSdfRecovery.cpp', 'SceneCreationRecovery.h', 'SceneCreationRecovery.cpp',
                     'RecoveryCaptureArchive.h', 'RecoveryCaptureArchive.cpp'):
            self.assertFalse((ROOT / 'source/live/scene' / name).exists(), name)
        for name in ('source/live/runtime/LiveLayerActor.h', 'source/live/runtime/LiveLayerAdmission.h',
                     'source/live/runtime/LiveLayerStatus.h', 'source/live/scene/SceneAttachment.h',
                     'source/live/scene/LiveSdfFileFormat.h', 'source/live/scene/LiveSdfFileFormat.cpp',
                     'source/live/scene/BoundLayer.h', 'source/live/scene/UsdLayerCreation.cpp',
                     'source/live/scene/UsdLayerBinding.h'):
            absent(self, ROOT / name, (r'\bcaptureRecovery\b', r'RecoveryCapture', r'\bLayerRecoveryProvider\b',
                                       r'\bUsdCreationRecoveryState\b', r'\bUsdInitializingTargetRecovery\b',
                                       r'\bbindRecoveryProvider\b', r'\bbindCreationRecovery\b', r'\bacquireRecovery\b',
                                       r'\bdetachForRecovery\b', r'\bfreezeInitializingCreationTarget\b',
                                       r'\bcaptureAttachedLiveRecovery\b', r'\brecoveryFrozen_'))


    def test_resync_and_quarantine_are_gone(self):
        for name in ('UsdHostResync.h', 'UsdHostResync.cpp'):
            self.assertFalse((ROOT / 'source/live/scene' / name).exists(), name)
        absent(self, ROOT / 'source/live/scene/HostLayerState.h', (r'\bResync\b', r'\bprepareResync\b', r'RecoveryData'))
        absent(self, ROOT / 'source/live/scene/HostSceneProjection.h', (r'\bHostMutationFault\b', r'\bexhausted\b'))

    RETIRED = ('source/live/scene/RecoveryArchive.h', 'source/live/scene/RecoveryArchive.cpp',
               'source/live/scene/RecoveryBundle.h', 'source/live/scene/RecoveryBundle.cpp',
               'source/live/scene/RecoveryBundleFile.h', 'source/live/scene/RecoveryBundleFile.cpp',
               'source/live/scene/RecoveryContainer.h', 'source/live/scene/RecoveryContainer.cpp',
               'source/live/scene/RecoveryData.h', 'source/live/scene/RecoveryData.cpp',
               'source/live/scene/RecoveryEditorRecord.h', 'source/live/scene/RecoveryEditorRecord.cpp',
               'source/live/scene/RecoveryFile.h', 'source/live/scene/RecoveryFile.cpp',
               'source/live/scene/RecoveryManifest.h', 'source/live/scene/RecoveryManifest.cpp',
               'source/live/scene/RecoveryScene.h', 'source/live/scene/RecoveryScene.cpp',
               'source/live/scene/RecoveryUsd.h', 'source/live/scene/RecoveryUsd.cpp',
               'source/live/scene/RecoveryInspect.cpp', 'source/live/scene/CreationRecovery.h',
               'source/live/scene/CreationRecovery.cpp', 'source/live/scene/CreationWorkflowRecovery.h',
               'source/live/scene/CreationWorkflowRecovery.cpp', 'source/live/scene/recovery-usd-plugInfo.json.in',
               'cmake/LiveRecoveryFile.cmake')

    def test_archive_sources_and_tooling_are_gone(self):
        for name in self.RETIRED:
            self.assertFalse((ROOT / name).exists(), name)
        for name in ('source/live/scene/plugInfo.json.in', 'source/live/scene/plugin/plugInfo.json.in',
                     'cmake/LivePlugin.cmake', 'cmake/LivePython.cmake', 'cmake/LiveWireScene.cmake',
                     'tools/live/build.py', 'tools/live/test.py'):
            absent(self, ROOT / name, (r'RecoveryResolver', r'live_recovery_inspect', r'recovery-usd', r'recovery.file'))

    def test_destroyed_data_transfer_and_capture_provenance_are_gone(self):
        scene = ROOT / 'source/live/scene'
        for name in ('HostSceneProjection.h', 'HostSceneProjection.cpp', 'HostLayerState.h', 'HostLayerState.cpp',
                     'LiveSdfFileFormat.cpp', 'BoundLayer.h', 'UsdLayerCreation.cpp'):
            absent(self, scene / name, (r'\bdetachDestroyedData\b', r'\bretirementSpare_', r'\bretired_\b',
                                        r'\bretired\(\)', r'\bisRetired\b', r'\bContentSource\b', r'\bHostCapture\b',
                                        r'(?i)storage retirement', r'\bnonretired\b'))

    def test_scene_comments_name_no_deleted_machinery(self):
        scene = ROOT / 'source/live/scene'
        absent(self, scene / 'UsdCreationRegistry.h', (r'(?i)recovery admission', r'(?i)export evidence'))
        absent(self, scene / 'UsdLayerBinding.h', (r'(?i)read views', r'(?i)\beditors\b'))
        absent(self, scene / 'SceneCreation.h', (r'(?i)workflow recovery',))
        absent(self, scene / 'SceneAttachment.h', (r'(?i)file-format integration requirements',))
        absent(self, scene / 'UsdLayerCreation.h', (r'(?i)creation API is provided here',))

if __name__ == '__main__':
    unittest.main()
