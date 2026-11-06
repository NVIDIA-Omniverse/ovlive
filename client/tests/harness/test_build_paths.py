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
    def test_debug_binary_maps_source_paths_and_records_only_the_pdb_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            project = base / 'checkout'
            project.mkdir()
            (project / 'probe.cpp').write_text('#include <cstdio>\nint main() { std::puts(__FILE__); }\n')
            (project / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.20)\nproject(paths LANGUAGES CXX)\n'
                'if(MAP_BUILD_PATHS)\n'
                f'include("{ROOT.as_posix()}/cmake/LiveBuildPaths.cmake")\nendif()\n'
                'add_executable(path_probe probe.cpp)\n')
            binaries = {}
            outputs = {}
            for mapped in (False, True):
                build = base / ('mapped' if mapped else 'control')
                configure = ['cmake', '-S', str(project), '-B', str(build), '-DCMAKE_BUILD_TYPE=Debug',
                             '-DMAP_BUILD_PATHS=' + ('ON' if mapped else 'OFF')]
                if shutil.which('ninja'):
                    configure += ['-G', 'Ninja']
                for command in (configure, ['cmake', '--build', str(build), '--config', 'Debug']):
                    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                paths = list(build.rglob('path_probe.exe' if os.name == 'nt' else 'path_probe'))
                self.assertEqual(len(paths), 1, paths)
                binary = binaries[mapped] = paths[0].read_bytes()
                done = subprocess.run([str(paths[0])], capture_output=True, timeout=15)
                self.assertEqual(done.returncode, 0, done.stderr)
                outputs[mapped] = done.stdout
                if os.name == 'nt':
                    marker = binary.index(b'RSDS')
                    pdb = binary[marker + 24:].split(b'\0', 1)[0]
                    if mapped:
                        self.assertEqual(pdb, b'path_probe.pdb')
                    else:
                        self.assertTrue(b'/' in pdb or b'\\' in pdb, pdb)
            spellings = {str(base).encode().lower(), base.as_posix().encode().lower()}
            self.assertTrue(any(root in outputs[False].lower() for root in spellings), 'control has no source path')
            for root in spellings:
                self.assertNotIn(root, outputs[True].lower())
                if os.name == 'nt':
                    self.assertFalse(root in binaries[True].lower(), 'Windows binary retains a build path')

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
