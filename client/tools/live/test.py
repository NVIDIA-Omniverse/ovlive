#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common live test entrypoint. Unimplemented suites are errors, never passes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live import sanitizer, stack_dump
from tools.live.foundation import MACOS_EXCLUSIONS
from tools.live.test_support import (containment_unsupported, parse_shard, process_cleanup_scope, record_seeds, redact, run_process,
                                     write_json)

# A component pass never stands in for the future complete product suites.
SUITES = {
    "foundation": [sys.executable, "-B", str(ROOT / "tools/live/foundation.py")],
    "uri": [sys.executable, "-m", "unittest", "discover", "-s", "tests/runtime", "-p", "test_uri.py", "-v"],
    "wire-envelope": [],  # OMPF framing/hash validation, not USD value decoding or scene application.
    "scene-reference": [],  # Actual scene/actor integration against the independent local service.
    "lock-noalloc": [],  # Allocation-fault contract for nested lock ownership; no SDK required.
    "scene-journal": [],  # Transport-free scene/intents publication; no live transport or product convergence.
    "wire-scene": [],  # Transport-free wire-to-scene transactions, not live convergence.
    "usd-values": [],  # Bounded USD value codec, not atomic scene application or live convergence.
    "sdf-scene": [],  # Transport-free real Sdf layer against stock USD; no network/plugin loading.
    "plugin-reference": [],  # Public shared plugin/host boundary against the local reference service.
    "writer-fuzz": [],  # Mandatory bounded multiprocess profiles; not the complete fuzz/release gate.
    "python-reference": [],  # Python ownership, finite requests, session streams and observer subscriptions.
    "kit-reference": [],  # Bounded actual headless Kit overlay/reference cases; not the full Kit release gate.
    "resolver-reference": [],  # Production resolver/runtime component against the local reference service.
    "resolver-probe": [],  # Test-only real USD resolver/format hook boundary; no service or production plugin.
    "diagnostics-schema": [],  # Offline schema validation of actual native CopyDiagnostics output.
    "runtime-foundation": [],  # Explicit native executable supplied by the caller.
    "reference-service": [],  # Explicit interpreter with pinned test dependencies.
    "native-reference": [],  # Native one-attempt RPC adapter against the reference service.
}
MACOS_SUITES = frozenset({"foundation", "uri", "wire-envelope", "lock-noalloc"})
# Each suite's package, tags, fixtures, contracts, resource budget and log redaction (docs/ovlive-testing.md).
CATALOG = json.loads((ROOT / "tests/contracts/suites.json").read_text(encoding="utf-8"))
# Reserved product suites: selecting one is an error, never a pass.
PLANNED = {name for name, entry in CATALOG["suites"].items() if entry.get("planned")}
# A reference lane's workload reports readiness once its reference service answers.
STARTUP_TIMEOUT_MS = 120000


def suite_platform_error(suite):
    """The supported supervised suite boundary, independent of how a suite is selected."""
    unsupported = containment_unsupported()
    if unsupported:
        return unsupported
    if sys.platform == "darwin" and suite in SUITES and suite not in MACOS_SUITES:
        return (f"{suite} is unsupported on macOS: supervised suite admission covers only "
                "foundation, uri, wire-envelope and lock-noalloc (open item platform)")
    return None


def fixture_fingerprint(suite):
    """SHA-256 over the path and bytes of every file of the suite's catalog fixtures."""
    digest = hashlib.sha256()
    for fixture in CATALOG["suites"][suite]["fixtures"]:
        base = ROOT / fixture
        files = [base] if base.is_file() else sorted(
            p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})
        for path in files:
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def scenario_fingerprint(suite, case_ids=None):
    """SHA-256 over the scenarios a suite covers: the recorded case IDs of a run, or, for a planned suite,
    the ID and requirement text of each contract the catalog assigns it."""
    if case_ids is not None:
        scenarios = sorted(case_ids)
    elif suite in PLANNED:
        contracts = {entry["id"]: entry["requirement"] for entry in
                     json.loads((ROOT / "tests/contracts/catalog.json").read_text(encoding="utf-8"))["contracts"]}
        scenarios = [[name, contracts[name]] for name in CATALOG["suites"][suite]["contracts"]]
    else:
        return None
    return hashlib.sha256(json.dumps(scenarios, sort_keys=True).encode()).hexdigest()


def select_suites(names, tags, packages):
    """The named suites, or every implemented suite that carries one of `tags` and belongs to one of `packages`."""
    if names is not None:
        return names.split(",")
    return [name for name, entry in CATALOG["suites"].items()
            if not entry.get("planned") and (not tags or set(tags) & set(entry["tags"]))
            and (not packages or entry["package"] in packages)]


def artifact_usage(directory):
    """Size of every file under the run directory, by relative path."""
    usage = {}
    for path in directory.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                usage[path.relative_to(directory).as_posix()] = path.stat().st_size
        except OSError:
            continue
    return usage


def apply_budget(result, suite, before, after):
    """Records the files and bytes a suite added to the run directory; exceeding its budget fails a passed suite."""
    added = {name: size for name, size in after.items() if before.get(name) != size}
    usage = {"artifact_bytes": sum(added.values()), "artifact_files": len(added)}
    budget = CATALOG["suites"][suite]["budget"]
    result["resource_usage"] = usage
    over = [key for key in ("artifact_bytes", "artifact_files") if usage[key] > budget[key]]
    if over and result["outcome"] == "passed":
        result.update(outcome="failed", reason="resource budget exceeded: " + ", ".join(
            f"{key} {usage[key]} > {budget[key]}" for key in over))
    return result


def redact_log(path, suite):
    """Applies the suite's fixture redaction to its bounded log; returns the replacement count."""
    patterns = [CATALOG["redaction"][name] for name in CATALOG["suites"][suite]["redact"]]
    if not patterns or not path.is_file():
        return 0
    text, count = redact(path.read_text(encoding="utf-8", errors="replace"), patterns)
    if count:
        path.write_text(text, encoding="utf-8")
    return count


def identity(command):
    """Nonempty tool output, or `unavailable: <ExceptionType>: <cause>` when it cannot be read."""
    try:
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, errors="replace", check=True, timeout=5)
    except subprocess.CalledProcessError as error:
        detail = next((line.strip() for line in (error.stderr or "").splitlines() if line.strip()), "")
        return f"unavailable: {type(error).__name__}: {error}" + (f": {detail[:200]}" if detail else "")
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {type(error).__name__}: {error}"
    return result.stdout.strip() or "unavailable: empty output"


def fingerprint():
    # A dirty working-tree hash, not just HEAD, is required for honest replay.
    paths = [ROOT / "include/omni/usd/live/Live.h", ROOT / "include/omni/usd/live/LiveUsd.h", ROOT / "CMakeLists.txt"]
    paths += sorted((ROOT / "source/live/runtime").glob("*.cpp"))
    paths += sorted((ROOT / "source/live/runtime").glob("*.h"))
    paths += sorted((ROOT / "source/live/runtime").glob("*.exports"))
    paths += sorted(p for directory in ("source/live/support", "source/live/wire", "source/live/scene", "source/live/resolver", "source/live/samples")
                    for p in (ROOT / directory).rglob("*") if p.is_file())
    paths += sorted((ROOT / "tests/runtime").glob("*.py"))
    paths += sorted((ROOT / "tests/reference_service").glob("*.py"))
    paths += sorted((ROOT / "tools/live").glob("*.py"))
    paths += sorted((ROOT / "tools/live").glob("*.supp"))
    paths += sorted((ROOT / "tests/harness").glob("*.py"))
    paths += sorted(p for p in (ROOT / "source/kit").rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})
    paths += sorted((ROOT / "tests/diagnostics").glob("*.py"))
    paths += sorted(p for p in (ROOT / "source/bindings/python/omni/usd/live").rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})
    paths += sorted((ROOT / "tests/native").rglob("*.c"))
    paths += sorted((ROOT / "tests/native").rglob("*.cpp"))
    paths += sorted((ROOT / "tests/native").rglob("*.h"))
    paths += sorted((ROOT / "tests/native").rglob("*.py"))
    paths += sorted((ROOT / "cmake").glob("*.cmake"))
    paths += sorted((ROOT / "source/live/protos").rglob("*.proto"))
    paths += sorted((ROOT / "tests/contracts").glob("*.json"))
    paths += sorted((ROOT / "tests/benchmarks").glob("*.json"))
    paths += [ROOT / "docs/ovlive-design.md"]
    # Provisioning pins are part of the tested cut-over too.
    paths += [ROOT / name for name in (
        "deps/live-services-target.packman.xml", "deps/live-services-host.packman.xml")]
    # One order on every platform: Windows paths sort case-insensitively, POSIX paths do not.
    paths.sort(key=lambda path: path.relative_to(ROOT).as_posix())
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def reference_complete(evidence, lane):
    """Reference-lane completion evidence: every discovered test ran and passed, none skipped."""
    return (isinstance(evidence, dict) and evidence.get("schema_version") == 1 and evidence.get("lane") == lane
            and evidence.get("passed") is True
            and type(evidence.get("discovered")) is int and evidence["discovered"] > 0
            and evidence.get("tests_run") == evidence["discovered"]
            and all(evidence.get(key) == 0 for key in
                    ("skipped", "failures", "errors", "expected_failures", "unexpected_successes")))


def unittest_ran_all(log):
    """A unittest command passes only when it ran at least one test and skipped none:
    one `Ran N tests` summary with N > 0 and a bare `OK` as the last non-empty line."""
    ran = re.findall(r"^Ran (\d+) tests? in ", log, re.M)
    lines = [line for line in log.splitlines() if line.strip()]
    return len(ran) == 1 and int(ran[0]) > 0 and bool(lines) and lines[-1] == "OK"


# Native component drivers end with one `live-cases passed=N` line naming their fixed case count.
NATIVE_CASES = {"wire-envelope": 6, "lock-noalloc": 2}
REFERENCE_LANES = {"reference-service", "native-reference", "scene-reference", "resolver-reference",
                   "plugin-reference", "python-reference", "kit-reference"}
CASE_OUTCOMES = {"passed", "failed", "error", "skipped", "expected_failure", "unexpected_success"}
# Suites whose evidence records one outcome per case; a passing one records exactly `tests_run` cases.
CASE_RECORDING = {"foundation", "writer-fuzz", "resolver-probe"} | REFERENCE_LANES


def record_sanitizer_reports(result, directory, index):
    """Lists a suite's ThreadSanitizer report files; any report fails a passed suite.

    halt_on_error=0 logs every report per process, so a report fails the suite even
    when a wrapper masks the process exit status.
    """
    reports = sorted(path.name for path in Path(directory).glob(f"{index}.tsan.*"))
    result["sanitizer_reports"] = reports
    if reports and result["outcome"] == "passed":
        result.update(outcome="failed", reason=f"ThreadSanitizer reported in {len(reports)} process(es)")
    return result


def artifact_names(paths, directory):
    """Stack dump paths relative to the run directory; a path outside it stays absolute."""
    names = []
    for path in paths:
        try:
            names.append(Path(path).relative_to(directory).as_posix())
        except ValueError:
            names.append(str(path))
    return names


def native_cases_ran(log, expected):
    """The last non-empty line is the only case-count line and reports the expected count."""
    lines = [line for line in log.splitlines() if line.strip()]
    return (sum(line.startswith("live-cases ") for line in lines) == 1
            and lines[-1] == f"live-cases passed={expected}")


def recorded_cases(suite, directory):
    """Per-case outcomes from a suite's evidence file, or None when the suite records none."""
    try:
        if suite == "foundation":
            cases = json.loads((directory / "foundation.json").read_text())["cases"]
        elif suite in REFERENCE_LANES:
            lane = "reference" if suite == "reference-service" else suite
            cases = json.loads((directory / f"{lane}-result.json").read_text())["cases"]
        elif suite == "writer-fuzz":
            results = json.loads((directory / "writer-fuzz/summary.json").read_text())["results"]
            cases = [{"id": entry["id"], "outcome": "passed" if entry["passed"] is True else "failed",
                      "message": entry.get("evidence_error", "")} for entry in results]
        elif suite == "resolver-probe":
            probes = json.loads((directory / "resolver-probe/creation/summary.json").read_text())["cases"]
            cases = [{"id": probe["name"], "outcome": "passed" if probe["passed"] is True else "failed",
                      "message": ""} for probe in probes]
        else:
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if (not isinstance(cases, list) or not cases or
            not all(isinstance(case, dict) and isinstance(case.get("id"), str) and
                    case.get("outcome") in CASE_OUTCOMES and isinstance(case.get("message"), str)
                    for case in cases)):
        return None
    return cases


def junit_suite(parent, suite, result, cases):
    """One testsuite per runner suite: its recorded cases, plus a suite-level case for a non-passing outcome."""
    node = ET.SubElement(parent, "testsuite", name=suite)
    failures = errors = 0
    for case in cases or []:
        element = ET.SubElement(node, "testcase", classname="live." + suite, name=case["id"])
        if case["outcome"] in {"failed", "unexpected_success"}:
            failures += 1
            ET.SubElement(element, "failure", type=case["outcome"]).text = case["message"]
        elif case["outcome"] == "error":
            errors += 1
            ET.SubElement(element, "error", type=case["outcome"]).text = case["message"]
        elif case["outcome"] in {"skipped", "expected_failure"}:
            ET.SubElement(element, "skipped", type=case["outcome"]).text = case["message"]
    if not cases or result["outcome"] != "passed":
        element = ET.SubElement(node, "testcase", classname="live." + suite, name=suite)
        if result["outcome"] == "failed":
            failures += 1
            ET.SubElement(element, "failure", type=result["outcome"]).text = json.dumps(result)
        elif result["outcome"] != "passed":
            errors += 1
            ET.SubElement(element, "error", type=result["outcome"]).text = json.dumps(result)
    node.set("tests", str(len(node)))
    node.set("failures", str(failures))
    node.set("errors", str(errors))
    return node


def prune_runs(artifact_dir, current, keep, sources=()):
    """Delete the oldest `run-*` directories beyond the `keep` newest earlier runs; never the current run
    and never a run that holds one of `sources` (the --replay or --writer-plan input)."""
    held = {parent for source in sources for parent in Path(source).resolve().parents}
    earlier = [path for path in artifact_dir.iterdir()
               if path.name.startswith("run-") and path != current and path.is_dir() and not path.is_symlink()
               and path.resolve() not in held]
    earlier.sort(key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    pruned = []
    for path in sorted(earlier[keep:], key=lambda path: path.name):
        try:
            shutil.rmtree(path)
        except OSError:
            continue
        pruned.append(path.name)
    return pruned


def diagnostic_schema_evidence(directory, expected_identity):
    """A zero exit alone is insufficient: require the lane's completed artifacts."""
    lane = directory / "diagnostics-schema"
    summary = json.loads((lane / "summary.json").read_text())
    validator_identity = json.loads((lane / "validator-identity.json").read_text())
    native_result = json.loads((lane / "native-result.json").read_text())
    if (summary.get("schema_version") != 1 or summary.get("boundary") != "diagnostic_snapshot" or
            summary.get("profile") != "currentProducer" or summary.get("passed") is not True or
            type(summary.get("snapshots")) is not int or not 0 < summary["snapshots"] <= 1024 or
            validator_identity != expected_identity or native_result.get("outcome") != "passed" or
            native_result.get("cleanup_complete") is not True or native_result.get("output_truncated") is not False):
        raise ValueError("invalid diagnostic schema completion evidence")
    with (lane / "snapshots.jsonl").open("rb") as source:
        corpus = source.read(64 * 1024 * 1024 + 1)
    if (not corpus or len(corpus) > 64 * 1024 * 1024 or summary.get("bytes") != len(corpus) or
            summary.get("sha256") != hashlib.sha256(corpus).hexdigest()):
        raise ValueError("diagnostic schema corpus hash/size mismatch")
    return summary


def test_support_inputs(root):
    """Shared native test support recorded by every test driver's manifest (cmake/LiveTestSupport.cmake)."""
    return {"cmake/LiveTestSupport.cmake",
            *(p.relative_to(root).as_posix() for p in (root / "tests/native/support").iterdir() if p.is_file())}


def native_identity(path, boundary="uri_component", *, source_root=None):
    """Reject a stale/missing binary or changed sources, including during replay."""
    if source_root is not None and boundary != "scene_benchmark":
        raise ValueError("alternate source roots are only supported for paired scene benchmarks")
    root = Path(source_root).resolve() if source_root is not None else ROOT
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["boundary"] != boundary:
        raise ValueError("unexpected native artifact manifest")
    if manifest.get("sanitizers") and boundary == "uri_component":
        raise ValueError("use the native sanitizer driver, not Python loading, for this artifact")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["artifact_sha256"]:
        raise ValueError("native artifact hash differs; rebuild required")
    required = {"CMakeLists.txt", "include/omni/usd/live/Live.h", "source/live/runtime/LiveUri.h",
                "source/live/runtime/LiveUri.cpp", "source/live/runtime/LiveUriApi.cpp", "source/live/runtime/Live.exports"}
    if boundary == "lock_noalloc":
        required = {"CMakeLists.txt", "cmake/LiveLockContract.cmake",
                    "source/live/support/ReaderWriterLock.h",
                    "source/live/support/ReaderWriterLock.inl",
                    "tests/native/core/lock_noalloc.cpp", "tools/live/write_build_manifest.py",
                    *test_support_inputs(root)}
    if boundary in {"runtime_foundation", "unary_transport"}:
        required |= {"source/live/runtime/LiveRuntime.h", "source/live/runtime/LiveRuntime.cpp", "source/live/support/LiveTsan.h",
                     "source/live/runtime/LiveRequest.h", "source/live/runtime/LiveRequest.cpp",
                     "source/live/runtime/LiveUnary.h", "source/live/runtime/LiveUnary.cpp",
                     "source/live/runtime/LiveDispatch.h", "source/live/runtime/LiveDispatch.cpp",
                     "source/live/runtime/LiveMetricDuration.h", "source/live/runtime/LiveDeadline.h", "source/live/runtime/LiveExceptionResult.h",
                     "source/live/runtime/LiveAuth.h", "source/live/runtime/LiveAuth.cpp",
                     "source/live/runtime/LiveFinite.cpp", "source/live/runtime/LiveMetadata.h",
                     "source/live/runtime/LiveLayerWrite.h", "source/live/runtime/LiveLayerWrite.cpp",
                     "source/live/runtime/LiveLayerRead.h", "source/live/runtime/LiveLayerRead.cpp",
                     "source/live/runtime/LiveLayerProgress.h", "source/live/runtime/LiveLayerProgress.cpp",
                     "source/live/runtime/LiveLayerActor.h", "source/live/runtime/LiveLayerActor.cpp",
                     "source/live/runtime/LiveJitter.h", "source/live/runtime/LiveJitter.cpp",
                     "source/live/runtime/LiveDrain.h", "source/live/runtime/LiveDrain.cpp",
                     "source/live/runtime/LiveStatus.h", "source/live/runtime/LiveStatus.cpp",
                     "source/live/runtime/LiveDiagnostics.h", "source/live/runtime/LiveDiagnostics.cpp",
                     "source/live/runtime/LiveDiagnosticsApi.cpp", "cmake/LiveBuildProvenance.h.in",
                     "source/live/runtime/LiveSession.h", "source/live/runtime/LiveSession.cpp",
                     "source/live/runtime/LiveSessionPeers.h", "source/live/runtime/LiveSessionPeers.cpp",
                     "source/live/runtime/LiveSessionApi.cpp",
                     "source/live/runtime/LiveLayerAdmission.h", "source/live/runtime/LiveLayerAdmission.cpp",
                     "source/live/runtime/LiveLayerStatus.h", "source/live/runtime/LiveLayerStatus.cpp",
                     "source/live/runtime/LiveRuntimeApi.cpp", "cmake/LiveGrpcDependencies.cmake",
                     "source/live/protos/nvidia/omniverse/live/v1alpha/live.proto"}
        required |= ({"tests/native/runtime/runtime_foundation.cpp", "tests/native/runtime/request_lifecycle.cpp",
                      "tests/contracts/diagnostic-catalog.json", "tools/live/diagnostic_catalog.py",
                      "tests/native/runtime/auth_controller.cpp", "tests/native/runtime/layer_write_lifecycle.cpp", "tests/native/runtime/layer_read_lifecycle.cpp", "tests/native/runtime/layer_progress.cpp", "tests/native/runtime/jitter.cpp", "tests/native/runtime/drain.cpp", "tests/native/runtime/status.cpp", "tests/native/runtime/diagnostics.cpp", "tests/native/runtime/diagnostics_api.cpp", "tests/native/runtime/session_peers.cpp", "tests/native/runtime/session_api.cpp", "tests/native/runtime/layer_admission.cpp", "tests/native/runtime/layer_status.cpp"}
                     if boundary == "runtime_foundation" else {"tests/native/runtime/unary_driver.cpp", "tests/native/runtime/layer_read_driver.cpp", "tests/native/runtime/layer_actor_driver.cpp", "tests/native/runtime/diagnostics_driver.cpp", "tests/native/runtime/session_driver.cpp"})
        required |= test_support_inputs(root)
        if not manifest.get("dependencies"):
            raise ValueError("runtime manifest is missing native dependency identity")
    if boundary == "scene_benchmark":
        if not manifest.get("compile_commands") or not any(Path(d["path"]).name == "compile_commands.json" for d in manifest.get("dependencies", [])):
            raise ValueError("benchmark manifest lacks compile-command provenance")
    if boundary in {"sdf_scene", "scene_benchmark", "usd_edit_boundary_probe"}:
        from tools.live.scene_identity import inputs
        required = set(inputs(root))
        if boundary == "usd_edit_boundary_probe":
            required |= {"tools/live/usd_edit_boundary.py", "tools/live/test.py", "tools/live/test_support.py",
                         "tools/live/process_guard.py", "tools/live/sanitizer.py"}
        if not manifest.get("dependencies") or len(manifest.get("dependency_trees", [])) != 3:
            raise ValueError("scene manifest is missing SDK identity")
    if boundary == "plugin":
        from tools.live.plugin_identity import inputs
        required = set(inputs(root))
    if boundary == "resolver_runtime":
        from tools.live.resolver_runtime_identity import inputs
        required = set(inputs(root))
    if boundary == "resolver_probe":
        from tools.live.resolver_identity import inputs
        required = set(inputs(root))
    if boundary == "wire_envelope":
        from tools.live.wire_identity import inputs
        required = set(inputs(root))
        if not manifest.get("flatbuffers") or len(manifest.get("dependency_trees", [])) != 1:
            raise ValueError("wire manifest is missing FlatBuffers SDK identity")
    if boundary in {"usd_values", "wire_scene", "scene_journal", "scene_transport"}:
        from tools.live.wire_identity import inputs, USD_TREES
        required = set(inputs(root, boundary))
        trees = manifest.get("dependency_trees", [])
        roles = {entry.get("role") for entry in trees}
        if (not manifest.get("flatbuffers") or len(trees) != 1 + len(USD_TREES) or
                roles != {"flatbuffers_include", *USD_TREES}):
            raise ValueError(f"{boundary} manifest is missing SDK identity")
    if boundary == "scene_transport" and not manifest.get("dependencies"):
        raise ValueError("scene transport manifest is missing native dependency identity")
    if boundary == "scene_transport":
        required.add("source/live/scene/plugInfo.json.in")
    if {entry["path"] for entry in manifest["inputs"]} != required:
        raise ValueError("native manifest is missing required source identity")
    for entry in manifest["inputs"]:
        source = (root / entry["path"]).resolve()
        source.relative_to(root.resolve())
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("native artifact has stale source inputs; rebuild required")
    for entry in manifest.get("dependencies", []):
        if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("native artifact has stale dependency inputs; rebuild required")
    if boundary == "scene_transport":
        metadata = path.resolve().parent / "native-file-format" / "plugInfo.json"
        if sum(Path(entry["path"]) == metadata for entry in manifest["dependencies"]) != 1:
            raise ValueError("scene transport manifest is missing expected file-format metadata identity")
        if metadata.resolve() != metadata:
            raise ValueError("scene transport file-format metadata is redirected")
        if metadata.read_bytes() != (root / "source/live/scene/plugInfo.json.in").read_bytes():
            raise ValueError("scene transport file-format metadata differs from tracked template")
    if boundary == "plugin":
        from tools.live.plugin_identity import validate_dependencies
        validate_dependencies(manifest)
    if boundary == "resolver_runtime":
        from tools.live.resolver_runtime_identity import validate_dependencies
        validate_dependencies(manifest)
    if boundary == "resolver_probe":
        from tools.live.resolver_identity import validate_dependencies
        validate_dependencies(manifest)
    if boundary in {"sdf_scene", "scene_benchmark", "usd_edit_boundary_probe"}:
        from tools.live.scene_identity import tree_hash
        for entry in manifest["dependency_trees"]:
            if tree_hash(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("native scene SDK differs; rebuild required")
    if boundary == "wire_envelope":
        from tools.live.wire_identity import sdk_identity, tree_hash
        entry = manifest["dependency_trees"][0]
        include = Path(entry["path"])
        if (tree_hash(include) != entry["sha256"] or
                sdk_identity(root, include) != manifest["flatbuffers"]):
            raise ValueError("native wire SDK differs; rebuild required")
    if boundary in {"usd_values", "wire_scene", "scene_journal", "scene_transport"}:
        from tools.live.wire_identity import sdk_identity, tree_hash
        for entry in manifest["dependency_trees"]:
            include = Path(entry["path"])
            if tree_hash(include) != entry["sha256"]:
                raise ValueError(f"native {boundary} SDK differs; rebuild required")
            if entry["role"] == "flatbuffers_include" and sdk_identity(root, include) != manifest["flatbuffers"]:
                raise ValueError(f"native {boundary} FlatBuffers identity differs; rebuild required")
    return manifest


def clean_environment():
    """Identity probing and execution share provisioned packages and host identity, excluding ambient credentials."""
    environment = {key: value for key, value in os.environ.items()
                   if key in {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "USERPROFILE"}
                   # Windows host identity and MSVC header/library discovery need these values.
                   or (sys.platform == "win32" and key in {"INCLUDE", "LIB", "LIBPATH",
                                                         "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432"})}
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONNOUSERSITE"] = "1"
    if sys.platform == "win32":
        # Text files default to UTF-8 as under a Linux UTF-8 locale, not to the Windows code page.
        environment["PYTHONUTF8"] = "1"
    return environment


def main():
    stack_dump.install()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", help="comma-separated suite names (default foundation)")
    parser.add_argument("--tag", help="comma-separated tags: select every implemented suite with one of them")
    parser.add_argument("--package", help="comma-separated packages: select every implemented suite of one of them")
    parser.add_argument("--scenario", help="comma-separated case IDs or dotted prefixes to run in case-selecting suites")
    parser.add_argument("--shard", help="I/N: run the cases whose ID hashes to shard I of N in case-selecting suites")
    parser.add_argument("--driver-mode", help="run one native driver mode as the only native-reference case")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument("--timeout-ms", type=int, help="failure watchdog; defaults to suite/profile budget")
    parser.add_argument("--max-output-bytes", type=int, default=1048576)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT.parent / ".cache/live-tests")
    parser.add_argument("--keep-runs", type=int, default=20,
                        help="earlier run-* directories to retain in the artifact directory (minimum 2)")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--writer-plan", type=Path,
                        help="replay a retained root-field writer plan (plugin-reference only)")
    parser.add_argument('--writer-fuzz-profile', choices=('ci', 'nightly', 'soak'), default='ci')
    parser.add_argument("--uri-library", type=Path, help="built URI contract probe; requires its build manifest")
    parser.add_argument("--wire-driver", type=Path, help="built OMPF envelope executable with pinned SDK manifest")
    parser.add_argument("--scene-transport-driver", type=Path, help="built scene transport executable with runtime/source/SDK manifest")
    parser.add_argument("--lock-driver", type=Path, help="built allocation-fault lock contract with source manifest")
    parser.add_argument("--scene-journal-driver", type=Path, help="built scene journal executable with source/SDK manifest")
    parser.add_argument("--wire-scene-driver", type=Path, help="built wire-to-scene executable with source/SDK manifest")
    parser.add_argument("--usd-values-driver", type=Path, help="built USD value codec executable with source/SDK manifest")
    parser.add_argument("--sdf-driver", type=Path, help="built Sdf scene executable with SDK manifest")
    parser.add_argument("--plugin-driver", type=Path, help="built public shared plugin driver with plugin/source/SDK manifest")
    parser.add_argument("--python-extension", type=Path, help="built _live extension with wrapper/plugin/SDK identity")
    parser.add_argument("--resolver-runtime-driver", type=Path, help="built production resolver/runtime component driver with plugin/SDK manifest")
    parser.add_argument("--resolver-driver", type=Path, help="built test-only USD resolver probe with plugin/SDK manifest")
    parser.add_argument("--diagnostic-python", type=Path, help="provisioned interpreter with pinned diagnostic schema dependencies")
    parser.add_argument("--runtime-driver", type=Path, help="built runtime foundation executable with build manifest")
    parser.add_argument("--unary-driver", type=Path, help="built native unary transport executable with build manifest")
    parser.add_argument("--reference-python", type=Path, help="provisioned Python with pinned reference-service dependencies")
    parser.add_argument("--reference-openssl", type=Path, help="pinned OpenSSL executable for local TLS fixtures")
    parser.add_argument("--list", action="store_true")
    from tools.live.kit_reference import INPUTS as KIT_INPUTS
    for name in KIT_INPUTS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path,
                            help="released packman package zip, beside its .sha256, that --kit-package was unpacked "
                                 "from; selects the release-package Kit lane" if name == "kit_release_package" else
                                 "explicit provisioned input for the headless Kit reference lane")
    args = parser.parse_args()
    if args.list:
        print("Available: " + ", ".join(sorted(SUITES)))
        print("Pending implementation: " + ", ".join(sorted(PLANNED)))
        for name, entry in CATALOG["suites"].items():
            print(f"  {name}: package {entry['package']}, tags {','.join(entry['tags'])}, "
                  f"fixtures {fixture_fingerprint(name)[:12]}"
                  + (f", scenarios {scenario_fingerprint(name)[:12]}" if name in PLANNED else ""))
        return 0
    if args.suite is not None and (args.tag or args.package):
        parser.error("--suite cannot combine with --tag or --package")
    tags = args.tag.split(",") if args.tag else []
    packages = args.package.split(",") if args.package else []
    known_tags = {tag for entry in CATALOG["suites"].values() for tag in entry["tags"]}
    known_packages = {entry["package"] for entry in CATALOG["suites"].values()}
    if set(tags) - known_tags or set(packages) - known_packages:
        parser.error("unknown tag or package: " + ", ".join(sorted((set(tags) - known_tags) | (set(packages) - known_packages))))
    selected = select_suites("foundation" if args.suite is None and not tags and not packages else args.suite,
                             tags, packages)
    if not selected:
        parser.error("the tag and package filters select no implemented suite")
    scenarios = args.scenario.split(",") if args.scenario is not None else []
    shard = args.shard
    driver_mode = args.driver_mode
    master_seed = args.seed if args.seed is not None else secrets.randbits(64)
    expected_fingerprint = None
    expected_native = None
    expected_wire = None
    expected_scene_transport = None
    expected_lock = None
    expected_scene_journal = None
    expected_wire_scene = None
    expected_usd_values = None
    expected_sdf = None
    expected_plugin = None
    expected_python_binding = None
    expected_kit = expected_kit_runtime = None
    expected_resolver_runtime = None
    expected_resolver = None
    expected_runtime = None
    expected_unary = None
    expected_reference = None
    expected_diagnostic_schema = None
    writer_replay = None
    expected_writer_context = None
    expected_fixtures = None
    if args.replay:
        try:
            sys.path.insert(0, str(ROOT / 'tests/reference_service'))
            from writer_history import read_json
            replay = read_json(args.replay, limit=16 * 1024 * 1024)
            if replay["schema_version"] != 1:
                raise ValueError("unsupported replay version")
            selected = replay["suites"]
            if "diagnostics-schema" in selected:
                args.diagnostic_python = Path(replay["diagnostic_python"])
                expected_diagnostic_schema = replay["diagnostic_schema_identity"]
                if not expected_diagnostic_schema:
                    raise ValueError("missing diagnostic validator replay identity")
            if "kit-reference" in selected:
                expected_kit = replay["kit_identity"]
                if not isinstance(expected_kit, dict) or not expected_kit:
                    raise ValueError("Kit replay requires selected-input identity")
                for name in KIT_INPUTS:
                    path = replay["kit_inputs"][name]
                    if path is not None and (not isinstance(path, str) or not Path(path).is_absolute()):
                        raise ValueError("invalid Kit replay input path")
                    setattr(args, name, Path(path) if path is not None else None)
                expected_kit_runtime = replay["kit_runtime"]
                if not isinstance(expected_kit_runtime, dict) or not expected_kit_runtime:
                    raise ValueError("Kit replay requires recorded runtime identity; original startup evidence is incomplete")
            master_seed = replay["master_seed"]
            scenarios = replay.get("scenarios") or []
            shard = replay.get("shard")
            driver_mode = replay.get("driver_mode")
            expected_fixtures = replay.get("fixture_fingerprints")
            args.clients = replay["clients"]
            args.timeout_ms = replay["timeout_ms"]
            args.max_output_bytes = replay["max_output_bytes"]
            expected_fingerprint = replay["source_fingerprint"]
            writer_replay = replay.get("writer_plan")
            if 'writer-fuzz' in selected:
                from tools.live.writer_campaign import replay_context
                expected_writer_context = replay_context(replay)
                args.writer_fuzz_profile = replay['writer_fuzz_profile']
                if args.writer_fuzz_profile not in ('ci', 'nightly', 'soak'):
                    raise ValueError('invalid writer fuzz replay profile')
            if replay.get("uri_library"):
                args.uri_library = Path(replay["uri_library"])
                expected_native = replay.get("native_identity")
            if replay.get("wire_driver"):
                args.wire_driver = Path(replay["wire_driver"])
                expected_wire = replay.get("wire_identity")
            if replay.get("scene_transport_driver"):
                args.scene_transport_driver = Path(replay["scene_transport_driver"])
                expected_scene_transport = replay.get("scene_transport_identity")
            if "lock-noalloc" in selected:
                if (not isinstance(replay.get("lock_driver"), str) or not replay["lock_driver"] or
                        not isinstance(replay.get("lock_identity"), dict) or not replay["lock_identity"]):
                    raise ValueError("lock-noalloc replay requires recorded driver and identity")
                args.lock_driver = Path(replay["lock_driver"])
                expected_lock = replay["lock_identity"]
            if replay.get("scene_journal_driver"):
                args.scene_journal_driver = Path(replay["scene_journal_driver"])
                expected_scene_journal = replay.get("scene_journal_identity")
            if replay.get("wire_scene_driver"):
                args.wire_scene_driver = Path(replay["wire_scene_driver"])
                expected_wire_scene = replay.get("wire_scene_identity")
            if replay.get("usd_values_driver"):
                args.usd_values_driver = Path(replay["usd_values_driver"])
                expected_usd_values = replay.get("usd_values_identity")
            if replay.get("sdf_driver"):
                args.sdf_driver = Path(replay["sdf_driver"])
                expected_sdf = replay.get("sdf_identity")
            if replay.get("plugin_driver"):
                args.plugin_driver = Path(replay["plugin_driver"])
                expected_plugin = replay.get("plugin_identity")
            if replay.get("python_extension"):
                args.python_extension = Path(replay["python_extension"])
                expected_python_binding = replay.get("python_binding_identity")
            if replay.get("resolver_runtime_driver"):
                args.resolver_runtime_driver = Path(replay["resolver_runtime_driver"])
                expected_resolver_runtime = replay.get("resolver_runtime_identity")
            if replay.get("resolver_driver"):
                args.resolver_driver = Path(replay["resolver_driver"])
                expected_resolver = replay.get("resolver_identity")
            if replay.get("runtime_driver"):
                args.runtime_driver = Path(replay["runtime_driver"])
                expected_runtime = replay.get("runtime_identity")
            if replay.get("unary_driver"):
                args.unary_driver = Path(replay["unary_driver"])
                expected_unary = replay.get("unary_identity")
            if replay.get("reference_python"):
                args.reference_python = Path(replay["reference_python"])
                expected_reference = replay.get("reference_identity")
                if replay.get("reference_openssl"):
                    args.reference_openssl = Path(replay["reference_openssl"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"invalid replay: {error}")
    if any(getattr(args, name) is not None for name in KIT_INPUTS) and "kit-reference" not in selected:
        parser.error("Kit inputs require --suite kit-reference")
    if args.timeout_ms is None:
        from tools.live.writer_campaign import PROFILES
        args.timeout_ms = max(PROFILES[args.writer_fuzz_profile]['campaign_timeout_ms'] if 'writer-fuzz' in selected else 60000,
                              150000 if 'kit-reference' in selected else 0)
    if (not selected or any(not isinstance(s, str) or not s for s in selected)
            or len(selected) != len(set(selected))):
        parser.error("select unique nonempty suite names")
    if master_seed < 0 or args.clients < 1 or args.timeout_ms <= 0 or args.max_output_bytes < 0:
        parser.error("invalid seed, client count, timeout, or output budget")
    if args.keep_runs < 2:
        parser.error("--keep-runs must retain at least 2 earlier runs")
    if scenarios or shard is not None:
        if any(not name for name in scenarios):
            parser.error("select nonempty scenario names")
        if shard is not None:
            try:
                parse_shard(shard)
            except ValueError as error:
                parser.error(str(error))
        without = [s for s in selected if not CATALOG["suites"].get(s, {}).get("case_selection")]
        if without:
            parser.error("--scenario and --shard need suites that select cases; not: " + ", ".join(without))
    if driver_mode is not None:
        if selected != ["native-reference"] or scenarios or shard is not None:
            parser.error("--driver-mode runs only --suite native-reference, without --scenario or --shard")
        if driver_mode not in CATALOG["suites"]["native-reference"]["driver_modes"]:
            parser.error("--driver-mode must be one of: " + ", ".join(CATALOG["suites"]["native-reference"]["driver_modes"]))
    if (scenarios or shard is not None) and args.writer_plan:
        parser.error("--writer-plan cannot combine with --scenario or --shard")
    if args.writer_plan and args.replay:
        parser.error("--writer-plan cannot override --replay")
    if args.writer_plan:
        sys.path.insert(0, str(ROOT / 'tests/reference_service'))
        from writer_history import load as load_writer_plan
        try:
            writer_replay = load_writer_plan(args.writer_plan)
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"invalid writer replay: {error}")
    if writer_replay is not None:
        if selected != ['plugin-reference']:
            parser.error("writer plan requires only --suite plugin-reference")
        sys.path.insert(0, str(ROOT / 'tests/reference_service'))
        from writer_history import validate as validate_writer_plan
        try:
            validate_writer_plan(writer_replay)
        except (ValueError, TypeError) as error:
            parser.error(f"invalid writer replay: {error}")
        master_seed = writer_replay['master_seed']
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.artifact_dir))
    source_hash = fingerprint()
    native = None
    native_error = None
    wire_native = None
    wire_error = None
    scene_transport_native = None
    scene_transport_error = None
    lock_native = None
    lock_error = None
    scene_journal_native = None
    scene_journal_error = None
    wire_scene_native = None
    wire_scene_error = None
    usd_values_native = None
    usd_values_error = None
    sdf_native = None
    sdf_error = None
    plugin_native = None
    python_binding = None
    python_binding_error = None
    kit_identity = kit_error = None
    plugin_error = None
    resolver_runtime_native = None
    resolver_runtime_error = None
    resolver_native = None
    resolver_error = None
    runtime_native = None
    runtime_error = None
    reference = None
    reference_error = None
    diagnostic_schema = None
    diagnostic_schema_error = None
    unary_native = None
    unary_error = None
    environment = clean_environment()
    admitted = {suite for suite in selected if not suite_platform_error(suite)}
    if "uri" in admitted:
        try:
            if args.uri_library is None:
                raise ValueError("--uri-library must identify the built URI contract probe")
            args.uri_library = args.uri_library.resolve()
            native = native_identity(args.uri_library)
            if expected_native is not None and native != expected_native:
                raise ValueError("native artifact differs from replay")
        except (OSError, ValueError, KeyError, TypeError) as error:
            native_error = str(error)
    if "wire-envelope" in admitted:
        try:
            if args.wire_driver is None:
                raise ValueError("--wire-driver must identify the built OMPF envelope executable")
            args.wire_driver = args.wire_driver.resolve()
            wire_native = native_identity(args.wire_driver, "wire_envelope")
            if expected_wire is not None and wire_native != expected_wire:
                raise ValueError("native wire artifact differs from replay")
        except (OSError, ValueError, KeyError, TypeError) as error:
            wire_error = str(error)
    if {"scene-reference", "writer-fuzz"}.intersection(admitted):
        try:
            if args.scene_transport_driver is None:
                raise ValueError("--scene-transport-driver must identify the built scene transport executable")
            args.scene_transport_driver = args.scene_transport_driver.resolve()
            scene_transport_native = native_identity(args.scene_transport_driver, "scene_transport")
            if expected_scene_transport is not None and scene_transport_native != expected_scene_transport:
                raise ValueError("native scene transport artifact differs from replay")
        except (OSError, ValueError, KeyError, TypeError) as error:
            scene_transport_error = str(error)
    if "lock-noalloc" in admitted:
        try:
            if args.lock_driver is None:
                raise ValueError("--lock-driver must identify the built allocation-fault lock executable")
            args.lock_driver = args.lock_driver.resolve()
            lock_native = native_identity(args.lock_driver, "lock_noalloc")
            if expected_lock is not None and lock_native != expected_lock:
                raise ValueError("native lock artifact differs from replay")
        except (OSError, ValueError, KeyError, TypeError) as error:
            lock_error = str(error)
    if "scene-journal" in admitted:
        try:
            if args.scene_journal_driver is None:
                raise ValueError("--scene-journal-driver must identify the built scene journal executable")
            args.scene_journal_driver = args.scene_journal_driver.resolve()
            scene_journal_native = native_identity(args.scene_journal_driver, "scene_journal")
            if expected_scene_journal is not None and scene_journal_native != expected_scene_journal:
                raise ValueError("native scene journal artifact differs from replay")
        except (OSError, ValueError, KeyError, TypeError) as error:
            scene_journal_error = str(error)
    if "wire-scene" in admitted:
        try:
            if args.wire_scene_driver is None:
                raise ValueError("--wire-scene-driver must identify the built wire-to-scene executable")
            args.wire_scene_driver = args.wire_scene_driver.resolve()
            wire_scene_native = native_identity(args.wire_scene_driver, "wire_scene")
            if expected_wire_scene is not None and wire_scene_native != expected_wire_scene:
                raise ValueError("native wire scene artifact differs from replay")
        except (OSError, ValueError, KeyError, TypeError) as error:
            wire_scene_error = str(error)
    if "usd-values" in admitted:
        try:
            if args.usd_values_driver is None:
                raise ValueError("--usd-values-driver must identify the built USD value codec executable")
            args.usd_values_driver = args.usd_values_driver.resolve()
            usd_values_native = native_identity(args.usd_values_driver, "usd_values")
            if expected_usd_values is not None and usd_values_native != expected_usd_values:
                raise ValueError("native USD values artifact differs from replay")
        except (OSError, ValueError, KeyError, TypeError) as error:
            usd_values_error = str(error)
    if "sdf-scene" in admitted:
        try:
            if args.sdf_driver is None:
                raise ValueError("--sdf-driver must identify the built Sdf executable")
            args.sdf_driver = args.sdf_driver.resolve()
            sdf_native = native_identity(args.sdf_driver, "sdf_scene")
            if expected_sdf is not None and sdf_native != expected_sdf:
                raise ValueError("native scene artifact differs from replay")
        except (OSError, ValueError, KeyError, TypeError) as error:
            sdf_error = str(error)
    if "kit-reference" in admitted:
        try:
            from tools.live.kit_reference import snapshot, validate_runtime
            kit_identity = snapshot(args)
            if expected_kit is not None and kit_identity != expected_kit:
                raise ValueError("Kit candidate or selected inputs differ from replay")
            if expected_kit_runtime is not None:
                validate_runtime(expected_kit_runtime, kit_identity)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            kit_error = str(error)
    if "python-reference" in admitted:
        try:
            from tools.live.python_identity import validate
            if args.python_extension is None:
                raise ValueError("--python-extension must identify the built Python module")
            args.python_extension = args.python_extension.resolve()
            python_binding = validate(args.python_extension)
            if expected_python_binding is not None and python_binding != expected_python_binding:
                raise ValueError("Python binding artifact differs from replay")
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            python_binding_error = str(error)
    if {"plugin-reference", "writer-fuzz"}.intersection(admitted):
        try:
            if args.plugin_driver is None:
                raise ValueError("--plugin-driver must identify the built shared plugin executable")
            args.plugin_driver = args.plugin_driver.resolve()
            plugin_native = native_identity(args.plugin_driver, "plugin")
            if expected_plugin is not None and plugin_native != expected_plugin:
                raise ValueError("native shared plugin artifact differs from replay")
        except (OSError, ValueError, KeyError, TypeError) as error:
            plugin_error = str(error)
    if "resolver-reference" in admitted:
        try:
            if args.resolver_runtime_driver is None:
                raise ValueError("--resolver-runtime-driver must identify the built resolver/runtime executable")
            args.resolver_runtime_driver = args.resolver_runtime_driver.resolve()
            resolver_runtime_native = native_identity(args.resolver_runtime_driver, "resolver_runtime")
            if expected_resolver_runtime is not None and resolver_runtime_native != expected_resolver_runtime:
                raise ValueError("native resolver/runtime artifact differs from replay")
        except (OSError, ValueError, KeyError, TypeError) as error:
            resolver_runtime_error = str(error)
    if "resolver-probe" in admitted:
        try:
            if args.resolver_driver is None:
                raise ValueError("--resolver-driver must identify the built USD resolver probe")
            args.resolver_driver = args.resolver_driver.resolve()
            resolver_native = native_identity(args.resolver_driver, "resolver_probe")
            if expected_resolver is not None and resolver_native != expected_resolver:
                raise ValueError("native resolver artifact differs from replay")
        except (OSError, ValueError, KeyError, TypeError) as error:
            resolver_error = str(error)
    if {"runtime-foundation", "diagnostics-schema"}.intersection(admitted):
        try:
            if args.runtime_driver is None:
                raise ValueError("--runtime-driver must identify the built runtime foundation executable")
            args.runtime_driver = args.runtime_driver.resolve()
            runtime_native = native_identity(args.runtime_driver, "runtime_foundation")
            if expected_runtime is not None and runtime_native != expected_runtime:
                raise ValueError("native runtime artifact differs from replay")
        except (OSError, ValueError, KeyError, TypeError) as error:
            runtime_error = str(error)
    if "diagnostics-schema" in admitted:
        try:
            if args.diagnostic_python is None:
                raise ValueError("--diagnostic-python must identify a provisioned validator interpreter")
            args.diagnostic_python = args.diagnostic_python.absolute()
            output = subprocess.run([str(args.diagnostic_python), "-B", str(ROOT / "tools/live/diagnostic_snapshot.py"), "--identity"],
                                    check=True, capture_output=True, text=True, timeout=15, env=environment)
            diagnostic_schema = json.loads(output.stdout)
            if expected_diagnostic_schema is not None and diagnostic_schema != expected_diagnostic_schema:
                raise ValueError("diagnostic validator identity differs from replay")
        except (OSError, ValueError, subprocess.SubprocessError):
            diagnostic_schema_error = "--diagnostic-python dependencies unavailable, mismatched, or different from replay"
    if "native-reference" in admitted:
        try:
            if args.unary_driver is None:
                raise ValueError("--unary-driver must identify the built unary transport executable")
            args.unary_driver = args.unary_driver.resolve()
            unary_native = native_identity(args.unary_driver, "unary_transport")
            if expected_unary is not None and unary_native != expected_unary:
                raise ValueError("native unary artifact differs from replay")
        except (OSError, ValueError, KeyError, TypeError) as error:
            unary_error = str(error)
    if {"reference-service", "native-reference", "scene-reference", "resolver-reference", "plugin-reference", "python-reference", "kit-reference", "writer-fuzz"}.intersection(admitted):
        try:
            if args.reference_python is None:
                raise ValueError("--reference-python must identify a provisioned test interpreter")
            if args.reference_openssl is None:
                raise ValueError("--reference-openssl must identify the pinned OpenSSL executable")
            # Resolving a venv executable symlink would discard its environment.
            args.reference_python = args.reference_python.absolute()
            args.reference_openssl = args.reference_openssl.resolve()
            output = subprocess.run([str(args.reference_python), str(ROOT / "tools/live/reference.py"), "--identity",
                                     "--openssl", str(args.reference_openssl)],
                                    check=True, capture_output=True, text=True, timeout=15, env=environment)
            reference = json.loads(output.stdout)
            if expected_reference is not None and reference != expected_reference:
                raise ValueError("reference-service dependency identity differs from replay")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            reference_error = str(error)

    # Children run with the options; evidence records their portable form.
    kinds = {sanitizer.kind_of(m)
             for m in (lock_native, plugin_native, resolver_runtime_native, resolver_native, scene_transport_native, scene_journal_native, wire_scene_native, usd_values_native, wire_native, sdf_native, runtime_native, unary_native)
             if m and m.get("sanitizers")}
    sanitizer_error = "mixed sanitizer kinds" if len(kinds) > 1 else None
    sanitizer_kind = next(iter(kinds), None)
    if (sanitizer_kind == "thread" and platform.system() == "Linux"
            and not sanitizer.address_randomization_disabled()):
        sanitizer_error = "ThreadSanitizer lanes run under setarch -R (GCC 12 libtsan cannot map memory under this kernel's ASLR)"
    sanitizer_options = sanitizer.runtime_options(sanitizer_kind, directory / "tsan") if sanitizer_kind else {}
    sanitizer_runtime = sanitizer.recorded(sanitizer_options)
    if expected_writer_context is not None:
        from tools.live.writer_campaign import canonical
        actual_writer_context = dict(source=source_hash, plugin=plugin_native, scene=scene_transport_native,
                                     reference=reference, sanitizer=sanitizer_runtime)
        if canonical(actual_writer_context) != canonical(expected_writer_context):
            plugin_error = 'writer campaign replay source, candidate or dependency identity differs'
    if writer_replay is not None:
        # Reject stale evidence before any native writer is launched. Ordinary
        # --replay additionally binds interpreter/dependency identities below.
        from scene_wire import empty_snapshot
        from writer_history import verify_provenance
        try:
            verify_provenance(writer_replay, source=source_hash, candidate=plugin_native,
                              reference=reference, sanitizer=sanitizer_runtime,
                              fixture=hashlib.sha256(empty_snapshot()).hexdigest())
        except ValueError as error:
            plugin_error = str(error)
        write_json(directory / 'writer.plan.json', writer_replay)
    fixture_fingerprints = {suite: fixture_fingerprint(suite) for suite in selected if suite in CATALOG["suites"]}
    replay = {"schema_version": 1, "suites": selected, "master_seed": master_seed,
              "scenarios": scenarios, "shard": shard, "driver_mode": driver_mode,
              "fixture_fingerprints": fixture_fingerprints,
              "clients": args.clients, "timeout_ms": args.timeout_ms,
              "max_output_bytes": args.max_output_bytes, "source_fingerprint": source_hash,
              "uri_library": str(args.uri_library) if args.uri_library else None, "native_identity": native,
              "wire_driver": str(args.wire_driver) if args.wire_driver else None, "wire_identity": wire_native,
              "scene_transport_driver": str(args.scene_transport_driver) if args.scene_transport_driver else None,
              "scene_transport_identity": scene_transport_native,
              "lock_driver": str(args.lock_driver) if args.lock_driver else None, "lock_identity": lock_native,
              "scene_journal_driver": str(args.scene_journal_driver) if args.scene_journal_driver else None,
              "scene_journal_identity": scene_journal_native,
              "wire_scene_driver": str(args.wire_scene_driver) if args.wire_scene_driver else None,
              "wire_scene_identity": wire_scene_native,
              "usd_values_driver": str(args.usd_values_driver) if args.usd_values_driver else None,
              "usd_values_identity": usd_values_native,
              "runtime_driver": str(args.runtime_driver) if args.runtime_driver else None,
              "runtime_identity": runtime_native,
              "sdf_driver": str(args.sdf_driver) if args.sdf_driver else None, "sdf_identity": sdf_native,
              "plugin_driver": str(args.plugin_driver) if args.plugin_driver else None,
              "plugin_identity": plugin_native,
              "python_extension": str(args.python_extension) if args.python_extension else None,
              "python_binding_identity": python_binding,
              "resolver_runtime_driver": str(args.resolver_runtime_driver) if args.resolver_runtime_driver else None,
              "resolver_runtime_identity": resolver_runtime_native,
              "resolver_driver": str(args.resolver_driver) if args.resolver_driver else None,
              "resolver_identity": resolver_native,
              "unary_driver": str(args.unary_driver) if args.unary_driver else None,
              "unary_identity": unary_native,
              "sanitizer_runtime": sanitizer_runtime,
              "reference_python": str(args.reference_python) if args.reference_python else None,
              "reference_openssl": str(args.reference_openssl) if args.reference_openssl else None,
              "reference_identity": reference, "writer_plan": writer_replay,
              "writer_fuzz_profile": args.writer_fuzz_profile,
              "diagnostic_schema_identity": diagnostic_schema,
              "diagnostic_python": str(args.diagnostic_python) if args.diagnostic_python else None}
    from tools.live.kit_reference import selection
    replay.update(kit_inputs=selection(args), kit_identity=kit_identity, kit_runtime=expected_kit_runtime)
    write_json(directory / "replay.json", replay)
    seeds = record_seeds(directory, master_seed, args.clients)
    manifest = {"schema_version": 1, "boundary": "foundation/URI/runtime/scene/Sdf/wire components; no full product qualification",
                "native_identity": native, "runtime_identity": runtime_native, "reference_identity": reference,
                "diagnostic_schema_identity": diagnostic_schema,
                "wire_identity": wire_native,
                "scene_transport_identity": scene_transport_native,
                "lock_identity": lock_native,
                "scene_journal_identity": scene_journal_native,
                "wire_scene_identity": wire_scene_native,
                "usd_values_identity": usd_values_native,
                "sdf_identity": sdf_native,
                "plugin_identity": plugin_native,
                "python_binding_identity": python_binding,
                "kit_identity": kit_identity,
                "resolver_runtime_identity": resolver_runtime_native,
                "resolver_identity": resolver_native,
                "unary_identity": unary_native,
                "sanitizer_runtime": sanitizer_runtime,
                "source_fingerprint": source_hash, "git_head": identity(["git", "rev-parse", "HEAD"]),
                "platform": platform.platform(), "python": sys.version, "seed": seeds,
                "process_cleanup": process_cleanup_scope(),
                "selection": {"scenarios": scenarios, "shard": shard, "driver_mode": driver_mode},
                "toolchain": {name: identity([name, "--version"]).splitlines()[0] for name in ("cc", "c++")},
                "pruned": [], "results": [], "complete": False}
    write_json(directory / "manifest.json", manifest)
    # Pruning follows input validation and the first manifest, so a rejected invocation deletes no evidence.
    manifest["pruned"] = prune_runs(args.artifact_dir, directory, args.keep_runs,
                                    [path for path in (args.replay, args.writer_plan) if path])
    write_json(directory / "manifest.json", manifest)
    xml = ET.Element("testsuites", name="live")
    failures = errors = 0
    environment.update(sanitizer_options)
    if args.uri_library:
        environment["LIVE_TEST_URI_LIBRARY"] = str(args.uri_library)
    environment["LIVE_TEST_SEED"] = str(master_seed)
    environment["LIVE_TEST_ARTIFACT_DIR"] = str(directory)
    environment["LIVE_TEST_SOURCE_FINGERPRINT"] = source_hash
    environment["LIVE_TEST_REFERENCE_IDENTITY"] = json.dumps(reference, sort_keys=True)
    interrupted = False
    for index, suite in enumerate(selected):
        try:
            platform_error = suite_platform_error(suite)
            if platform_error:
                result = {"outcome": "infrastructure_error", "reason": platform_error}
            elif sanitizer_error:
                result = {"outcome": "infrastructure_error", "reason": sanitizer_error}
            elif expected_fingerprint and expected_fingerprint != source_hash:
                result = {"outcome": "infrastructure_error", "reason": "replay source fingerprint differs"}
            elif expected_fixtures is not None and expected_fixtures.get(suite) != fixture_fingerprints.get(suite):
                result = {"outcome": "infrastructure_error", "reason": "replay fixture fingerprint differs"}
            elif suite == "uri" and native_error:
                result = {"outcome": "infrastructure_error", "reason": native_error}
            elif suite == "wire-envelope" and wire_error:
                result = {"outcome": "infrastructure_error", "reason": wire_error}
            elif suite in {"scene-reference", "writer-fuzz"} and scene_transport_error:
                result = {"outcome": "infrastructure_error", "reason": scene_transport_error}
            elif suite == "lock-noalloc" and lock_error:
                result = {"outcome": "infrastructure_error", "reason": lock_error}
            elif suite == "scene-journal" and scene_journal_error:
                result = {"outcome": "infrastructure_error", "reason": scene_journal_error}
            elif suite == "wire-scene" and wire_scene_error:
                result = {"outcome": "infrastructure_error", "reason": wire_scene_error}
            elif suite == "usd-values" and usd_values_error:
                result = {"outcome": "infrastructure_error", "reason": usd_values_error}
            elif suite == "sdf-scene" and sdf_error:
                result = {"outcome": "infrastructure_error", "reason": sdf_error}
            elif suite == "kit-reference" and kit_error:
                result = {"outcome": "infrastructure_error", "reason": kit_error}
            elif suite == "python-reference" and python_binding_error:
                result = {"outcome": "infrastructure_error", "reason": python_binding_error}
            elif suite in {"plugin-reference", "writer-fuzz"} and plugin_error:
                result = {"outcome": "infrastructure_error", "reason": plugin_error}
            elif suite == "resolver-reference" and resolver_runtime_error:
                result = {"outcome": "infrastructure_error", "reason": resolver_runtime_error}
            elif suite == "resolver-probe" and resolver_error:
                result = {"outcome": "infrastructure_error", "reason": resolver_error}
            elif suite == "diagnostics-schema" and diagnostic_schema_error:
                result = {"outcome": "infrastructure_error", "reason": diagnostic_schema_error}
            elif suite in {"runtime-foundation", "diagnostics-schema"} and runtime_error:
                result = {"outcome": "infrastructure_error", "reason": runtime_error}
            elif suite == "native-reference" and unary_error:
                result = {"outcome": "infrastructure_error", "reason": unary_error}
            elif suite in {"reference-service", "native-reference", "scene-reference", "resolver-reference", "plugin-reference", "python-reference", "kit-reference", "writer-fuzz"} and reference_error:
                result = {"outcome": "infrastructure_error", "reason": reference_error}
            elif suite not in SUITES:
                reason = "suite not implemented" if suite in PLANNED else "unknown suite"
                result = {"outcome": "infrastructure_error", "reason": reason}
            else:
                suite_environment = environment
                command = [str(args.runtime_driver)] if suite == "runtime-foundation" else SUITES[suite]
                if suite == 'writer-fuzz':
                    from tools.live.writer_campaign import generate as writer_campaign
                    campaign_context = dict(source=source_hash, plugin=plugin_native, scene=scene_transport_native,
                                            reference=reference, sanitizer=sanitizer_runtime)
                    campaign_plan = writer_campaign(args.writer_fuzz_profile, master_seed)
                    campaign_plan['context'] = campaign_context
                    context_path = directory / 'writer-fuzz-context.json'
                    write_json(context_path, campaign_context)
                    command = [str(args.reference_python), '-B', str(ROOT / 'tools/live/writer_fuzz.py'),
                               '--profile', args.writer_fuzz_profile, '--plugin-driver', str(args.plugin_driver),
                               '--scene-driver', str(args.scene_transport_driver), '--openssl', str(args.reference_openssl),
                               '--context', str(context_path)]
                if suite == "diagnostics-schema":
                    command = [str(args.diagnostic_python), "-B", str(ROOT / "tools/live/diagnostic_schema_suite.py"),
                               str(args.runtime_driver), str(args.timeout_ms), str(args.max_output_bytes)]
                if suite == "wire-envelope":
                    command = [str(args.wire_driver), str(master_seed)]
                if suite == "scene-journal":
                    command = [str(args.scene_journal_driver), str(master_seed)]
                if suite == "wire-scene":
                    command = [str(args.wire_scene_driver), str(master_seed)]
                if suite == "usd-values":
                    command = [str(args.usd_values_driver), str(master_seed)]
                if suite == "sdf-scene":
                    command = [str(args.sdf_driver), str(master_seed)]
                if suite == "resolver-probe":
                    # Keep the single-handler test process independent of ambient
                    # plugin discovery; do not change other suites' environments.
                    suite_environment = dict(environment, PXR_PLUGINPATH_NAME="",
                                             PXR_AR_DEFAULT_SEARCH_PATH="", AR_DEFAULT_RESOLVER="")
                    probe_directory = directory / "resolver-probe"
                    probe_directory.mkdir()
                    plugin_info = next(entry["path"] for entry in resolver_native["dependencies"]
                                       if entry["role"] == "plug_info")
                    command = [sys.executable, "-B", str(ROOT / "tests/native/resolver/creation.py"),
                               str(args.resolver_driver), plugin_info, str(probe_directory), "--with-baseline"]
                if suite == "lock-noalloc":
                    command = [str(args.lock_driver)]
                if suite in {"reference-service", "native-reference", "scene-reference", "resolver-reference", "plugin-reference", "python-reference", "kit-reference"}:
                    command = [str(args.reference_python), str(ROOT / "tools/live/reference.py"),
                               "--openssl", str(args.reference_openssl)]
                    if suite == "plugin-reference":
                        plug_info = next(entry["path"] for entry in plugin_native["dependencies"]
                                         if entry["role"] == "plug_info")
                        sample = next(e["path"] for e in plugin_native["dependencies"] if e["role"] == "diagnostics_sample")
                        command += ["--plugin-driver", str(args.plugin_driver), "--plugin-plug-info", plug_info,
                                    "--diagnostics-sample", sample]
                        if writer_replay is not None:
                            command += ['--writer-plan', str(directory / 'writer.plan.json')]
                        suite_environment = dict(environment, PXR_PLUGINPATH_NAME="",
                                                 PXR_AR_DEFAULT_SEARCH_PATH="", AR_DEFAULT_RESOLVER="")
                    if suite == "kit-reference":
                        from tools.live.kit_reference import arguments, reference_environment
                        command += arguments(kit_identity)
                        suite_environment = reference_environment(environment, kit_identity)
                    if suite == "python-reference":
                        command += ["--python-extension", str(args.python_extension)]
                        sdk = python_binding["plugin_identity"]["dependency_trees"]
                        libraries = [e["path"] for e in sdk if e["role"] in {"usd_lib", "python_lib"}]
                        usd_python = [str(Path(e["path"]) / "python") for e in sdk if e["role"] == "usd_lib"]
                        if sys.platform == "win32":
                            # Windows finds DLLs on PATH; oneTBB's DLLs are in the USD bin directory.
                            search = [*libraries, *(str(Path(path).parent / "bin") for path in libraries)]
                            library_path = dict(PATH=os.pathsep.join([*search, environment.get("PATH", "")]))
                        else:
                            library_path = dict(LD_LIBRARY_PATH=os.pathsep.join(libraries))
                        suite_environment = dict(environment, **library_path,
                                                 PYTHONPATH=os.pathsep.join([str(ROOT), *usd_python]),
                                                 PXR_PLUGINPATH_NAME="", PXR_AR_DEFAULT_SEARCH_PATH="",
                                                 AR_DEFAULT_RESOLVER="", PYTHONDONTWRITEBYTECODE="1")
                    if suite == "resolver-reference":
                        plugin = next(entry["path"] for entry in resolver_runtime_native["dependencies"]
                                      if entry["role"] == "plugin")
                        plug_info = next(entry["path"] for entry in resolver_runtime_native["dependencies"]
                                         if entry["role"] == "plug_info")
                        command += ["--resolver-runtime-driver", str(args.resolver_runtime_driver),
                                    "--resolver-runtime-plugin", plugin, "--resolver-runtime-plug-info", plug_info]
                        suite_environment = dict(environment, PXR_PLUGINPATH_NAME="",
                                                 PXR_AR_DEFAULT_SEARCH_PATH="", AR_DEFAULT_RESOLVER="")
                    if suite == "scene-reference":
                        command += ["--scene-transport-driver", str(args.scene_transport_driver)]
                    if suite == "native-reference":
                        command += ["--native-driver", str(args.unary_driver)]
                        if driver_mode is not None:
                            command += ["--driver-mode", driver_mode]
                if sanitizer_kind == "thread":
                    suite_environment = dict(suite_environment,
                                             **sanitizer.runtime_options("thread", directory / f"{index}.tsan"))
                if scenarios or shard is not None:
                    suite_environment = dict(suite_environment, LIVE_TEST_SCENARIOS=json.dumps(scenarios),
                                             LIVE_TEST_SHARD=shard or "")
                before = artifact_usage(directory)
                result = asdict(run_process(command, cwd=ROOT,
                                output=directory / f"{index}.log", timeout_ms=args.timeout_ms,
                                max_output_bytes=args.max_output_bytes, environment=suite_environment,
                                ready=directory / f"{index}.ready.json" if suite in REFERENCE_LANES else None,
                                startup_timeout_ms=STARTUP_TIMEOUT_MS if suite in REFERENCE_LANES else None,
                                scratch=True))
                if result["outcome"] == "service_start_failed":
                    result["reason"] = "reference service did not start: " + result["start_failure"]
                result["redactions"] = redact_log(directory / f"{index}.log", suite)
                result["stack_dumps"] = artifact_names(result["stack_dumps"], directory)
                if sanitizer_kind == "thread":
                    record_sanitizer_reports(result, directory, index)
                if suite == "kit-reference":
                    from tools.live.kit_reference import release_sha256
                    result["kit_lane"] = kit_identity["lane"]
                    result["kit_release_package_sha256"] = release_sha256(kit_identity)
                    # Preserve runtime identity for failed assertions too, when
                    # all three children initialized and wrote their identity.
                    # A startup failure without that evidence cannot promise a
                    # pinned-runtime replay; its recorded original command remains.
                    try:
                        from tools.live.kit_reference import participants, snapshot
                        runtime = participants(directory, kit_identity, require_clean=False)
                        if snapshot(args) == kit_identity and fingerprint() == source_hash:
                            if expected_kit_runtime is not None and runtime != expected_kit_runtime:
                                result.update(outcome="infrastructure_error", reason="Kit loaded runtime differs from replay")
                            else:
                                replay["kit_runtime"] = runtime
                                write_json(directory / "replay.json", replay)
                    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                        pass
                if suite == "kit-reference" and result["outcome"] == "passed":
                    try:
                        from tools.live.kit_reference import completion, snapshot
                        summary = completion(directory, kit_identity, source_hash, master_seed)
                        if snapshot(args) != kit_identity or fingerprint() != source_hash:
                            raise ValueError("Kit inputs or sources changed during common-runner execution")
                        if expected_kit_runtime is not None and summary["participants"] != expected_kit_runtime:
                            raise ValueError("Kit loaded runtime differs from replay")
                        replay["kit_runtime"] = summary["participants"]
                        write_json(directory / "replay.json", replay)
                        result["tests_run"] = summary["tests_run"]
                        result["kit_profile"] = summary["profile"]
                    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                        result.update(outcome="infrastructure_error", reason="Invalid Kit completion evidence: " + str(error))
                if suite == 'foundation' and result['outcome'] == 'passed':
                    try:
                        summary = json.loads((directory / 'foundation.json').read_text())
                        if (summary.get('schema_version') != 1 or summary.get('boundary') != 'foundation'
                                or summary.get('passed') is not True or type(summary.get('discovered')) is not int
                                or summary['discovered'] <= 0 or summary.get('tests_run') != summary['discovered']
                                or summary.get('skipped') != 0 or summary.get('expected_failures') != 0
                                or len(set(summary['test_ids'])) != summary['discovered']):
                            raise ValueError('Incomplete foundation evidence')
                        exclusions = [dict(id=identity, reason=reason) for identity, reason in MACOS_EXCLUSIONS.items()] if sys.platform == 'darwin' else []
                        if (summary.get('excluded', []) != exclusions
                                or set(summary['test_ids']) & {case['id'] for case in exclusions}):
                            raise ValueError('Unexpected foundation exclusions')
                        result['tests_run'] = summary['tests_run']
                        result['excluded'] = exclusions
                    except (OSError, ValueError, KeyError, TypeError):
                        result.update(outcome='infrastructure_error', reason='Foundation completion evidence invalid')
                if suite in NATIVE_CASES and result['outcome'] == 'passed':
                    expected_line = f"live-cases passed={NATIVE_CASES[suite]}"
                    if not native_cases_ran((directory / f"{index}.log").read_text(errors="replace"), NATIVE_CASES[suite]):
                        result.update(outcome='infrastructure_error', reason=f'{suite} log lacks final "{expected_line}"')
                if suite == 'uri' and result['outcome'] == 'passed':
                    if not unittest_ran_all((directory / f"{index}.log").read_text(errors="replace")):
                        result.update(outcome='infrastructure_error', reason='URI suite ran no tests or skipped tests')
                if (suite in {'reference-service', 'native-reference', 'scene-reference', 'resolver-reference',
                              'plugin-reference', 'python-reference'} and result['outcome'] == 'passed'):
                    lane = 'reference' if suite == 'reference-service' else suite
                    try:
                        evidence = json.loads((directory / (lane + '-result.json')).read_text())
                        if not reference_complete(evidence, lane):
                            raise ValueError('incomplete reference run')
                        result['tests_run'] = evidence['tests_run']
                    except (OSError, ValueError, KeyError, TypeError):
                        result.update(outcome='infrastructure_error', reason='Reference completion evidence invalid')
                if suite == 'writer-fuzz' and result['outcome'] == 'passed':
                    try:
                        from tools.live.writer_campaign import verify_summary
                        sys.path.insert(0, str(ROOT / 'tests/reference_service'))
                        from writer_history import read_json
                        summary = read_json(directory / 'writer-fuzz/summary.json', limit=16 * 1024 * 1024)
                        verify_summary(campaign_plan, summary)
                        result['writer_fuzz'] = dict(profile=args.writer_fuzz_profile, cases=len(summary['results']))
                        result['tests_run'] = len(summary['results'])
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        result.update(outcome='infrastructure_error', reason=f'writer campaign evidence invalid: {error}')
                if suite == "diagnostics-schema" and result["outcome"] == "passed":
                    try:
                        result["diagnostic_schema"] = diagnostic_schema_evidence(directory, diagnostic_schema)
                    except (OSError, ValueError, KeyError, TypeError):
                        result["outcome"] = "infrastructure_error"
                        result["reason"] = "diagnostic schema completion evidence missing or invalid"
                if suite == "resolver-probe" and result["outcome"] == "passed":
                    try:
                        trace_path = probe_directory / "trace.json"
                        trace = json.loads(trace_path.read_text())
                        if (trace["schema_version"] != 1 or trace["boundary"] != "usd_resolver_hook_probe" or
                                trace["usd_version"] != 2511 or trace["passed"] is not True or not trace["steps"]):
                            raise ValueError("invalid resolver probe trace")
                        result["trace"] = str(trace_path)
                        contract_path = probe_directory / "creation/summary.json"
                        contract = json.loads(contract_path.read_text())
                        required = {
                            "create-deferred-work": ("queued-anonymous-work", "outer-create-returned", "before-deferred-anonymous", "after-deferred-anonymous"),
                            "create-deferred-save-notice": ("save-notice-queued", "queued-anonymous-work", "outer-create-returned", "before-deferred-anonymous", "after-deferred-anonymous"),
                            "create-release-failure": ("outer-create-returned", "failed-target-destroyed", "before-find-after-failure", "after-find-after-failure"),
                            "create-freeze-failure": ("after-initial-data-swap", "outer-create-returned", "failed-target-destroyed", "frozen-backing-survived-target", "before-find-after-failure", "after-find-after-failure"),
                            "create-install": ("before-initial-data-swap", "after-initial-data-swap", "outer-create-returned"),
                            "export-nested": ("before-nested-anonymous", "after-nested-anonymous"),
                        }
                        cases = contract["cases"]
                        if (contract["schema_version"] != 1 or contract["boundary"] != "usd_creation_hook_contract" or
                                contract["usd_version"] != 2511 or contract["passed"] is not True or
                                contract["service"] is not False or contract["production_plugin"] is not False or
                                len(cases) != len(required) or {case["name"] for case in cases} != set(required)):
                            raise ValueError("invalid creation hook contract")
                        for probe_case in cases:
                            markers = probe_case["markers"]
                            positions = [markers.index(marker) for marker in required[probe_case["name"]]]
                            expected_hook = "WriteToFile" if probe_case["name"] == "export-nested" else "SaveToFile"
                            if (probe_case["passed"] is not True or probe_case["timed_out"] is not False or
                                    probe_case["output_limited"] is not False or probe_case["returncode"] != 0 or
                                    not {"anonymous-control-passed", "content-notice-control-passed", "completed"} <= set(markers) or
                                    positions != sorted(set(positions)) or expected_hook not in probe_case["hooks"]):
                                raise ValueError("invalid creation hook events")
                        result["creation_contract"] = str(contract_path)
                        result["tests_run"] = len(cases)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        result = {"outcome": "infrastructure_error", "reason": f"resolver trace missing or invalid: {error}"}
                apply_budget(result, suite, before, artifact_usage(directory))
        except KeyboardInterrupt:
            interrupted = True
            result = {"outcome": "interrupted", "reason": "user interrupted test; child cleanup requested"}
        cases = recorded_cases(suite, directory)
        if (result["outcome"] == "passed" and suite in CASE_RECORDING
                and (cases is None or len(cases) != result.get("tests_run"))):
            result = {"outcome": "infrastructure_error", "reason": "case count differs from tests run"}
        if suite in CATALOG["suites"]:
            result["fixture_fingerprint"] = fixture_fingerprints[suite]
            result["scenario_fingerprint"] = scenario_fingerprint(
                suite, [case["id"] for case in cases] if cases is not None else None)
        if result["outcome"] == "failed":
            failures += 1
        elif result["outcome"] != "passed":
            errors += 1
        junit_suite(xml, suite, result, cases)
        manifest["results"].append({"suite": suite, **result})
        write_json(directory / "manifest.json", manifest)
        print(f"{suite}: {result['outcome']}", flush=True)
        if interrupted:
            break
    manifest["complete"] = not interrupted
    manifest["passed"] = not failures and not errors and not interrupted
    write_json(directory / "manifest.json", manifest)
    for key in ("tests", "failures", "errors"):
        xml.set(key, str(sum(int(node.get(key)) for node in xml)))
    ET.ElementTree(xml).write(directory / "junit.xml", encoding="utf-8", xml_declaration=True)
    print(f"Artifacts: {directory}")
    # The hint names this script by its absolute path: it runs from any working directory, on any Windows drive.
    print(f"Replay: {sys.executable} {Path(__file__).resolve()} --replay {directory / 'replay.json'}")
    return 130 if interrupted else 2 if errors else 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
