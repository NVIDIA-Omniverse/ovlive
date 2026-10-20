# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The build maps every dependency package's resolved directory out of __FILE__ strings."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which('cmake'), 'cmake is not on PATH')
class BuildPathTests(unittest.TestCase):
    def test_linked_dependency_package_maps_to_its_symbolic_name(self):
        # GCC names a system header by its symlink-resolved path when that path is shorter, so a packman link
        # into a package cache needs a map of its own, not only the dependency root's. packman links a package
        # with a symlink on Linux and a junction on Windows, which CMake's REAL_PATH does not resolve.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = base / 'c/grpc/1'
            (package / 'include').mkdir(parents=True)
            deps = base / 'checkout/deps-links/target-deps'
            deps.mkdir(parents=True)
            if os.name == 'nt':
                import _winapi
                _winapi.CreateJunction(str(package), str(deps / 'grpc-release'))
            else:
                os.symlink(package, deps / 'grpc-release')
            project = base / 'project'
            project.mkdir()
            # CMake strings take forward slashes on every platform; a backslash is an escape.
            (project / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.20)\nproject(paths NONE)\n'
                f'include("{ROOT.as_posix()}/cmake/LiveBuildPaths.cmake")\n'
                'get_directory_property(options COMPILE_OPTIONS)\n'
                'message(STATUS "options=${options}")\n')
            result = subprocess.run(['cmake', '-S', str(project), '-B', str(base / 'build'),
                                     f'-DLIVE_DEPENDENCY_ROOT={deps.as_posix()}'],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            if os.name == 'nt':
                # MSVC names a header by the include directory it was found in, the link, which keeps its own map.
                self.assertIn(f'-ffile-prefix-map={(deps / "grpc-release").as_posix()}=<deps>/grpc-release',
                              result.stdout)
            else:
                self.assertIn(f'-ffile-prefix-map={package.resolve().as_posix()}=<deps>/grpc-release', result.stdout)
            self.assertIn(f'-ffile-prefix-map={deps.as_posix()}=<deps>', result.stdout)


if __name__ == '__main__':
    unittest.main()
