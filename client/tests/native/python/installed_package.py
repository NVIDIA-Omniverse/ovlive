# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check actual CMake installation against the production package allowlist."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.live import build, platforms

SAMPLES = (("live_diagnostics_sample", "Usage: live_diagnostics_sample "),
           ("live_participant_sample", "live_participant_sample --authority "))
PLUGIN_LIBRARY_PATH = ("@loader_path" if sys.platform == "darwin" else "$ORIGIN") + "/../lib/omni_usd_live/lib"


def run_samples(directory, environment, cwd):
    for name, usage in SAMPLES:
        result = subprocess.run([str(directory / "bin" / name), "--help"], check=True,
                                env=environment, cwd=cwd, text=True, capture_output=True, timeout=5)
        assert result.stdout.startswith(usage), result.stdout
        assert not result.stderr, result.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--usd-root", required=True)
    parser.add_argument("--config", default="Release")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="live-installed-package-") as directory:
        scratch = Path(directory).resolve()
        install, package = scratch / "install", scratch / "package"
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        subprocess.run([args.cmake, "--install", args.build_dir, "--config", args.config,
                        "--prefix", str(install)], check=True, env=environment, cwd=scratch, timeout=15)
        # The install prefix keeps the plugin under lib/omni_usd_live/lib; each
        # sample in bin/ must find it through its own search path.
        for name, _ in SAMPLES:
            if sys.platform == "darwin":
                dynamic = subprocess.check_output(["otool", "-l", str(install / "bin" / name)], text=True)
                paths = re.findall(r"cmd LC_RPATH\s+cmdsize \d+\s+path (.*?) \(offset", dynamic)
                assert PLUGIN_LIBRARY_PATH in paths, (name, paths)
            else:
                dynamic = subprocess.check_output(["readelf", "-d", str(install / "bin" / name)], text=True)
                paths = re.findall(r"\((?:RUNPATH|RPATH)\).*\[([^\]]*)\]", dynamic)
                assert len(paths) == 1 and PLUGIN_LIBRARY_PATH in paths[0].split(":"), (name, paths)
        run_samples(install, environment, scratch)
        build.assemble_install(install, package)
        actual = {p.relative_to(package).as_posix() for p in package.rglob("*")
                  if p.is_file() or p.is_symlink()}
        expected = set(build.package_files(platforms.host().name))
        assert actual == expected, {"unexpected": sorted(actual - expected),
                                    "missing": sorted(expected - actual)}
        assert not any(p.is_symlink() for p in package.rglob("*"))
        build._validate_layout(package, platforms.host().name, exact=True)
        # Reuse the real import/load-order checks with installed, relocated bytes.
        subprocess.run([sys.executable, "-B", str(ROOT / "tests/native/python/package.py"),
                        str(package), args.usd_root], check=True, env=environment, cwd=scratch, timeout=45)
        run_samples(package, environment, scratch)
        print(f"Actual installed payload: {len(actual)} allowlisted files; "
              "installed and relocated samples and imports passed")


if __name__ == "__main__":
    main()
