# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""C ABI declaration and consumer-layout probes; these do not exercise a runtime."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class AbiDraftTests(unittest.TestCase):
    def test_c_and_cpp_consumers_accept_extensible_record_pointer_arrays(self):
        source = ROOT / "tests/native/abi/header_contract.c"
        for language, compiler, standard in (("c", "cc", "c11"), ("c++", "c++", "c++17")):
            with self.subTest(language=language), tempfile.TemporaryDirectory() as temporary:
                executable = shutil.which("cl" if os.name == "nt" else compiler)
                self.assertIsNotNone(executable, f"required {language} toolchain unavailable")
                output = Path(temporary) / ("header-contract.exe" if os.name == "nt" else "header-contract")
                if os.name == "nt":
                    command = [executable, "/nologo", "/TC" if language == "c" else "/TP", f"/std:{standard}",
                               "/W4", "/WX", "/permissive-", "/I", str(ROOT / "include"), str(source),
                               f"/Fe{output}", f"/Fo{Path(temporary)}\\"]
                else:
                    command = [executable, "-x", language, f"-std={standard}", "-Wall", "-Wextra", "-Werror",
                               "-pedantic", "-I", str(ROOT / "include"), str(source), "-o", str(output)]
                compiled = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                          timeout=30)
                self.assertEqual(compiled.returncode, 0, compiled.stdout)
                subprocess.run([str(output)], check=True, timeout=5)

    def test_all_designed_exports_declared_and_no_removed_api(self):
        design = (ROOT / "docs/ovlive-design.md").read_text()
        table = design.split("| Area | Function |", 1)[1].split("Deliberately delete", 1)[0]
        expected = set()
        for line in table.splitlines():
            if line.startswith("| "):
                field = line.split("|")[2]
                expected.update("omniUsdLive" + name for name in re.findall(r"`([A-Z][A-Za-z]+)", field))
        header = (ROOT / "include/omni/usd/live/Live.h").read_text()
        declarations = set(re.findall(r"OMNI_USD_LIVE_CALL (omniUsdLive\w+)\(", header))
        self.assertEqual(declarations, expected)
        export_map = (ROOT / "source/live/runtime/Live.exports").read_text()
        self.assertEqual(set(re.findall(r"(omniUsdLive\w+);", export_map)), declarations)
        for removed in ("omniUsdLiveProcess", "omniUsdLiveProcessUpTo", "omniUsdLiveGetServerTime",
                        "omniUsdLiveGetLatestServerTime", "omniUsdLiveRegisterWakeCallback"):
            self.assertNotIn(removed, declarations)
        self.assertNotIn("#include <pxr", header)
        self.assertNotIn("#include <grpc", header)
        self.assertNotIn("OmniClient", header)


if __name__ == "__main__":
    unittest.main()
