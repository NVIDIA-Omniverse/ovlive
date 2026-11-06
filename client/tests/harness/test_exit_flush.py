# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Driver and sample stdout survives a Windows process exit that skips the C runtime's exit-time flush (issue #48)."""
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "tests/native/exit_flush"
RECORD = b"x" * 13000 + b" end\n"


class ExitFlushTests(unittest.TestCase):
    def test_assertions_and_termination_reach_stderr_and_still_fail(self):
        compiler = shutil.which("cl" if os.name == "nt" else "c++")
        self.assertIsNotNone(compiler, "required native toolchain unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "failure_report.cpp"
            source.write_text('#include <cassert>\n#include <new>\n#include <cstring>\n'
                              'int main(int, char** argv) { if (std::strcmp(argv[1], "assert") == 0) '
                              '{ assert(false && "crt-assert-probe"); } else { throw std::bad_alloc(); } }\n')
            executable = directory / "failure_report.exe"
            support = str(ROOT / "tests/native/support/StackDump.cpp")
            if os.name == "nt":
                command = [compiler, "/nologo", "/MDd", "/EHsc", "/std:c++17", "/UNDEBUG",
                           source.name, support, "dbghelp.lib", f"/Fe{executable}", f"/Fo{directory}\\"]
                # _wassert writes UTF-16LE to the binary stderr stream; the terminate handler writes narrow text.
                markers = (("assert", "crt-assert-probe".encode("utf-16-le")), ("throw", b"native test terminated"))
            else:
                command = [compiler, "-std=c++17", "-pthread", "-UNDEBUG", source.name, support, "-o", str(executable)]
                markers = (("assert", b"crt-assert-probe"), ("throw", b"std::bad_alloc"))
            built = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            for mode, marker in markers:
                done = subprocess.run([str(executable), mode], stdin=subprocess.DEVNULL,
                                      capture_output=True, timeout=15)
                self.assertNotEqual(done.returncode, 0, mode)
                self.assertIn(marker, done.stderr, (mode, done.stderr))
                if os.name == "nt" and mode == "throw":
                    self.assertIn(b"native test failure stack:", done.stderr)
            if os.name == 'nt':
                child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read(1)'], stdin=subprocess.PIPE)
                try:
                    dump = directory / 'waiting-child.dmp'
                    collector = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/live/windows_dump.py'),
                                                str(child.pid), str(dump)], capture_output=True, timeout=20)
                    self.assertEqual(collector.returncode, 0, collector.stdout + collector.stderr)
                    content = dump.read_bytes()
                    self.assertEqual(content[:4], b'MDMP')
                    streams, offset = struct.unpack_from('<II', content, 8)
                    directory_entries = [struct.unpack_from('<III', content, offset + 12 * i) for i in range(streams)]
                    threads = next(entry for entry in directory_entries if entry[0] == 3)
                    self.assertGreater(struct.unpack_from('<I', content, threads[2])[0], 0)
                finally:
                    child.kill()
                    child.communicate(timeout=15)

    def build(self, directory):
        sources = [str(PROBE / "exit_flush.cpp"), str(ROOT / "tests/native/support/StackDump.cpp")]
        include = str(ROOT / "source/live/samples/diagnostics")
        if os.name == "nt":
            # /MD as every native driver and sample builds: the C runtime is the ucrtbase DLL.
            compiler = shutil.which("cl")
            self.assertIsNotNone(compiler, "required MSVC toolchain unavailable")
            commands = [[compiler, "/nologo", "/MD", "/W4", "/WX", str(PROBE / "exit_lock.cpp"), "/LD",
                         f"/Fe{directory / 'live_exit_lock.dll'}", f"/Fo{directory}\\"],
                        [compiler, "/nologo", "/MD", "/EHsc", "/std:c++17", "/W3", "/I", include, *sources,
                         str(directory / "live_exit_lock.lib"), "dbghelp.lib", f"/Fe{directory / 'exit_flush.exe'}",
                         f"/Fo{directory}\\"]]
            executable = directory / "exit_flush.exe"
        else:
            compiler = shutil.which("c++")
            self.assertIsNotNone(compiler, "required C++ toolchain unavailable")
            executable = directory / "exit_flush"
            commands = [[compiler, "-std=c++17", "-Wall", "-Wextra", "-pthread", "-I", include, *sources,
                         "-o", str(executable)]]
        for command in commands:
            built = subprocess.run(command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   timeout=120)
            self.assertEqual(built.returncode, 0, built.stdout)
        return executable

    def test_driver_and_sample_records_survive_an_exit_that_skips_the_crt_flush(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = self.build(Path(temporary))
            outputs = {}
            for mode in ("crt", "driver", "sample"):
                # Failure-only watchdog: a passing run ends as soon as main returns.
                try:
                    done = subprocess.run([str(executable), mode], stdout=subprocess.PIPE, timeout=60)
                except subprocess.TimeoutExpired as expired:
                    self.fail(f"{mode}: the reproduction no longer ends the process at the orphaned lock; Windows or "
                              f"UCRT exit behavior changed. Captured {len(expired.stdout or b'')} bytes of stdout: "
                              f"{(expired.stdout or b'')[-64:]!r}")
                self.assertEqual(done.returncode, 0, mode)
                outputs[mode] = done.stdout
        self.assertEqual(outputs["driver"], RECORD)
        self.assertEqual(outputs["sample"], RECORD)
        crt = outputs["crt"]
        if os.name == "nt":
            # The process ended before the C runtime flushed: only whole buffers of the probe's 4 KiB stdout buffer
            # arrived, as in the evidence of issue #48, and the exit code still reports success.
            self.assertTrue(RECORD.startswith(crt))
            self.assertLess(len(crt), len(RECORD))
            self.assertEqual(len(crt) % 4096, 0)
        else:
            # glibc flushes stdio in exit() itself, so nothing ends the process before the flush.
            self.assertEqual(crt, RECORD)


if __name__ == "__main__":
    unittest.main()
