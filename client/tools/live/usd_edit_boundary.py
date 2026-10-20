#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce pinned USD edit-boundary limitations; never a product safety pass."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live import sanitizer
from tools.live.test import native_identity
from tools.live.test_support import run_process, write_json


OBSERVATIONS = {
    "permission": dict(readonly_data_calls=0, override_permission=True, override_data_calls=1),
    "move_control": dict(apply_returned_success=True, data_move_calls=1, old_data_present=False,
                         new_data_present=True, retained_path="/Q", retained_dormant=False),
    "noop_move": dict(apply_returned_success=True, data_move_calls=1, old_data_present=True,
                      new_data_present=False, retained_path="/Q", retained_dormant=True),
    "delegate": dict(notifications=1, permission_after_callback=False,
                     data_set_calls=1, value_was_written=True),
}


def observation(mode, output, sdk_lib):
    """Fail on changed SDK behavior or broken controls, rather than infer safety."""
    records = [json.loads(line) for line in output.splitlines()]
    if len(records) != 2:
        raise ValueError("expected one SDK identity and one observation")
    identity, observed = records
    if (identity.get("schema_version") != 1 or identity.get("usd_version") != 2511 or
            identity.get("boundary") != "usd_edit_boundary_diagnostic"):
        raise ValueError("unexpected SDK diagnostic identity")
    # JSON comparison distinguishes booleans from integers (unlike dict equality).
    expected = dict(mode=mode, **OBSERVATIONS[mode])
    if json.dumps(observed, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError(f"SDK behavior or control changed: {observed!r}")
    library = Path(identity["sdf_library"])
    if not library.is_absolute():
        raise ValueError("loaded Sdf library has no absolute identity")
    library.resolve().relative_to(sdk_lib.resolve())
    if library.name not in {"libov_25.11usd_ms.so", "libusd_sdf.so"}:
        raise ValueError("unexpected loaded Sdf library")
    return dict(observed=observed, sdf_library=str(library),
                sdf_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.artifact_dir))
    report = dict(schema_version=1, boundary="usd_edit_boundary_diagnostic",
                  product_qualified=False, complete=False, results=[])
    write_json(directory / "summary.json", report)
    print(f"SDK diagnostic artifacts: {directory}", flush=True)
    try:
        driver = args.driver.resolve()
        before = native_identity(driver, "usd_edit_boundary_probe")
        report["candidate"] = before
        sdk_lib = Path(before["dependency_trees"][1]["path"])
        python_lib = Path(before["dependencies"][0]["path"]).parent
        environment = {key: value for key, value in os.environ.items()
                       if key in {"PATH", "TMPDIR", "TEMP", "TMP"}}
        environment.update(LD_LIBRARY_PATH=os.pathsep.join((str(sdk_lib), str(python_lib))),
                           PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        if before["sanitizers"]:
            environment.update(sanitizer.runtime_options(sanitizer.kind_of(before), directory / "tsan"))
        for mode in OBSERVATIONS:
            command = [str(driver), mode]
            entry = dict(mode=mode, command=command)
            report["results"].append(entry)
            write_json(directory / "summary.json", report)
            log = directory / (mode + ".log")
            process = asdict(run_process(command, cwd=ROOT, output=log, timeout_ms=15000,
                                         max_output_bytes=65536, environment=environment))
            entry["process"] = process
            if (process["outcome"] != "passed" or process["cleanup_complete"] is not True or
                    process["output_truncated"] is not False):
                raise ValueError("SDK diagnostic process did not complete cleanly: " + mode)
            entry.update(observation(mode, log.read_text(), sdk_lib))
            write_json(directory / "summary.json", report)
        if native_identity(driver, "usd_edit_boundary_probe") != before:
            raise ValueError("diagnostic source, binary or SDK changed during execution")
        if len({item["sdf_library_sha256"] for item in report["results"]}) != 1:
            raise ValueError("cases loaded different Sdf libraries")
        report.update(complete=True, pinned_limitations_reproduced=True)
        print("Pinned SDK limitations reproduced with valid controls; product safety remains unqualified.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report["error"] = str(error)
        print(f"SDK diagnostic incomplete: {error}", file=sys.stderr)
        return 1
    finally:
        write_json(directory / "summary.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
