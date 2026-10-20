# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Driver and sample stdout survives a Windows process exit that skips the C runtime's exit-time flush (issue #48)."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "tests/native/exit_flush"
RECORD = b"x" * 13000 + b" end\n"


class ExitFlushTests(unittest.TestCase):
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
                         str(directory / "live_exit_lock.lib"), f"/Fe{directory / 'exit_flush.exe'}",
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
