# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package metadata rejects competing runtimes beyond direct ELF imports."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('live_plugin_package_audit', ROOT / 'tests/native/plugin/package.py')
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


class PluginPackageAuditTests(unittest.TestCase):
    def test_elf_checks_forbidden_names_in_each_dynamic_metadata_field(self):
        for field in ('NEEDED', 'SONAME', 'RPATH', 'RUNPATH'):
            for name in ('libOmniClient.so', 'resolver_runtime_component.so'):
                with self.subTest(field=field, name=name), \
                        patch.object(package.sys, 'platform', 'linux'), \
                        patch.object(package.subprocess, 'check_output',
                                     return_value=f' 0x00000001 ({field}) metadata: [{name}]\n'):
                    with self.assertRaises(AssertionError):
                        package.check_dependency_metadata(Path('fixture.so'))

    def test_elf_allows_host_sdk_dependencies_and_package_paths(self):
        report = (' (NEEDED) Shared library: [libusd_sdf.so]\n'
                  ' (SONAME) Library soname: [libomni_usd_live.so]\n'
                  ' (RUNPATH) Library runpath: [$ORIGIN]\n')
        with patch.object(package.sys, 'platform', 'linux'), \
                patch.object(package.subprocess, 'check_output', return_value=report):
            package.check_dependency_metadata(Path('fixture.so'))

    def test_macho_checks_imports_without_screening_the_artifact_filename(self):
        report = ('/fixture/omniclient/libomni_usd_live.dylib:\n'
                  '\t@rpath/libusd_sdf.dylib (compatibility version 0.0.0, current version 0.0.0)\n')
        with patch.object(package.sys, 'platform', 'darwin'), \
                patch.object(package.subprocess, 'check_output', return_value=report):
            package.check_dependency_metadata(Path('fixture.dylib'))
        with patch.object(package.sys, 'platform', 'darwin'), \
                patch.object(package.subprocess, 'check_output', return_value=report.replace('libusd_sdf', 'libOmniClient')):
            with self.assertRaises(AssertionError):
                package.check_dependency_metadata(Path('fixture.dylib'))
