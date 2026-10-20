#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build wire bindings and run the independent reference-service contract suite.

The caller selects an already-provisioned interpreter. No implicit installation
or network fetch. --identity works without socket permissions.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
PINS = {"grpcio": "1.78.0", "grpcio-tools": "1.78.0", "protobuf": "6.33.6"}


def identity(openssl):
    packages = {}
    for name, expected in PINS.items():
        distribution = importlib.metadata.distribution(name)
        if distribution.version != expected:
            raise RuntimeError(f"{name} requires {expected}, found {distribution.version}")
        digest = hashlib.sha256()
        for entry in sorted(distribution.files or [], key=str):
            if str(entry).endswith((".py", ".so", ".pyd", "/METADATA")):
                path = Path(distribution.locate_file(entry))
                digest.update(str(entry).encode() + b"\0" + path.read_bytes())
        packages[name] = {"version": expected, "files_sha256": digest.hexdigest()}
    openssl = openssl.resolve()
    version = subprocess.run([str(openssl), "version"], check=True, capture_output=True,
                             text=True, timeout=5).stdout.strip()
    if not version.startswith("OpenSSL 3.5.9 "):
        raise RuntimeError(f"reference TLS fixture requires OpenSSL 3.5.9, found {version}")
    return {"schema_version": 1, "boundary": "reference service", "python": sys.version,
            "interpreter": str(Path(sys.executable).absolute()), "packages": packages,
            "openssl": {"path": str(openssl), "version": version,
                        "sha256": hashlib.sha256(openssl.read_bytes()).hexdigest()},
            # The TLS proxy's stack is this interpreter's OpenSSL, not the pinned executable.
            "python_ssl": ssl.OPENSSL_VERSION}


def complete_run(result, discovered):
    """A lane passes only when every discovered test ran and passed: no skip,
    expected failure or unexpected success, and at least one test."""
    return (result.wasSuccessful() and discovered > 0 and result.testsRun == discovered and not result.skipped
            and not result.expectedFailures and not result.unexpectedSuccesses)


async def serve_one_ping(seed):
    """Readiness: one reference service starts on loopback, answers a Ping through a real channel and stops."""
    import grpc
    from service import Principal, ReferenceServer, State, pb, rpc
    state = State(seed)
    state.tokens = {"synthetic-readiness": Principal("readiness")}
    server = ReferenceServer(state)
    address = await server.start()
    try:
        async with grpc.aio.insecure_channel(address) as channel:
            await rpc.LiveServiceStub(channel).Ping(pb.PingRequest(), metadata=(("authorization", "Bearer synthetic-readiness"),),
                                                    timeout=10)  # Failure-only bound.
    finally:
        await server.stop()


def main():
    """Runs the lane; a lane that ends before its service answered reports a failed start."""
    sys.path.insert(0, str(ROOT))
    from tools.live.test_support import report_ready
    try:
        return lane_main()
    except BaseException as error:
        report_ready(False, f"{type(error).__name__} before the service answered")  # No-op once ready.
        raise


def lane_main():
    from tools.live import stack_dump
    from tools.live.test_support import case_selection, report_ready, select_cases
    stack_dump.install()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--openssl", type=Path, required=True)
    drivers = parser.add_mutually_exclusive_group()
    drivers.add_argument("--native-driver", type=Path)
    drivers.add_argument("--scene-transport-driver", type=Path)
    drivers.add_argument("--resolver-runtime-driver", type=Path)
    drivers.add_argument("--plugin-driver", type=Path)
    drivers.add_argument("--python-extension", type=Path)
    drivers.add_argument("--kit-executable", type=Path)
    parser.add_argument("--kit-package", type=Path)
    parser.add_argument("--kit-client-extension", type=Path)
    parser.add_argument("--kit-resolver-extension", type=Path)
    parser.add_argument("--kit-library-dir", type=Path)
    parser.add_argument("--kit-core-extension", type=Path)
    parser.add_argument("--kit-release-package", type=Path)
    parser.add_argument("--plugin-plug-info", type=Path)
    parser.add_argument("--diagnostics-sample", type=Path)
    parser.add_argument("--writer-plan", type=Path)
    parser.add_argument("--writer-case", type=Path)
    parser.add_argument("--resolver-runtime-plugin", type=Path)
    parser.add_argument("--resolver-runtime-plug-info", type=Path)
    parser.add_argument("--driver-mode", help="run only this native driver mode (requires --native-driver)")
    args = parser.parse_args()
    if not __debug__:
        parser.error("reference qualification requires enabled Python assertions")
    kit_inputs = (args.kit_executable, args.kit_package, args.kit_client_extension, args.kit_resolver_extension)
    if any(kit_inputs) and not all(kit_inputs):
        parser.error("Kit execution needs executable, package, client and resolver extension paths")
    if args.kit_library_dir and not args.kit_executable:
        parser.error("--kit-library-dir requires --kit-executable")
    if args.kit_core_extension and not args.kit_executable:
        parser.error("--kit-core-extension requires --kit-executable")
    if args.kit_release_package and not args.kit_executable:
        parser.error("--kit-release-package requires --kit-executable")
    if args.writer_plan and not args.plugin_driver:
        parser.error('--writer-plan requires --plugin-driver')
    if args.writer_case and (args.writer_plan or not (args.plugin_driver or args.scene_transport_driver)):
        parser.error('--writer-case requires one plugin/scene driver and cannot combine with --writer-plan')
    resolver_options = (args.resolver_runtime_driver, args.resolver_runtime_plugin, args.resolver_runtime_plug_info)
    if any(resolver_options) and not all(resolver_options):
        parser.error("--resolver-runtime-driver, --resolver-runtime-plugin and --resolver-runtime-plug-info must be supplied together")
    plugin_inputs = (args.plugin_driver, args.plugin_plug_info, args.diagnostics_sample)
    if any(plugin_inputs) and not all(plugin_inputs):
        parser.error("--plugin-driver, --plugin-plug-info and --diagnostics-sample must be supplied together")
    if args.driver_mode and not args.native_driver:
        parser.error("--driver-mode requires --native-driver")
    writer_case = None
    try:
        scenarios, shard = case_selection()
        if (scenarios or shard or args.driver_mode) and (args.kit_executable or args.writer_plan or args.writer_case):
            raise ValueError("case selection and driver modes are unavailable for Kit, writer plans and writer cases")
        dependencies = identity(args.openssl)
        if args.identity:
            print(json.dumps(dependencies, sort_keys=True))
            return 0
        if args.kit_executable:
            sys.path.insert(0, str(ROOT))
            from tools.live.kit_reference import snapshot, participants, release_sha256, PROFILE, TESTS
            from tools.live.test import fingerprint
            kit_identity = snapshot(args)
            kit_source = fingerprint()
            dependencies['kit_identity'] = kit_identity
            dependencies['source_fingerprint'] = kit_source
            for name, path in zip(('EXECUTABLE', 'PACKAGE', 'CLIENT_EXTENSION', 'RESOLVER_EXTENSION'), kit_inputs):
                resolved = path.resolve()
                if not resolved.exists():
                    raise ValueError('Missing Kit input: ' + name)
                os.environ['LIVE_TEST_KIT_' + name] = str(resolved)
            os.environ.pop('LIVE_TEST_KIT_LIBRARY_DIR', None)
            os.environ.pop('LIVE_TEST_KIT_CORE_EXTENSION', None)
            if args.kit_core_extension:
                os.environ['LIVE_TEST_KIT_CORE_EXTENSION'] = str(args.kit_core_extension.resolve())
            if args.kit_library_dir:
                if not args.kit_library_dir.is_dir():
                    raise ValueError('Kit runtime library directory is missing')
                os.environ['LIVE_TEST_KIT_LIBRARY_DIR'] = str(args.kit_library_dir.resolve())
        if args.plugin_driver:
            # The direct entrypoint has the same replay checks as the common
            # runner. Do this before even starting the reference service.
            sys.path[:0] = [str(ROOT), str(ROOT / 'tests/reference_service')]
            from tools.live.sanitizer import recorded as sanitizer_record
            from tools.live.test import fingerprint, native_identity
            from writer_history import load as load_writer_plan, verify_provenance
            from scene_wire import empty_snapshot
            source = fingerprint()
            candidate = native_identity(args.plugin_driver.resolve(), 'plugin')
            for role, selected in (('plug_info', args.plugin_plug_info),
                                   ('diagnostics_sample', args.diagnostics_sample)):
                recorded = next(entry['path'] for entry in candidate['dependencies'] if entry['role'] == role)
                if selected.resolve() != Path(recorded).resolve():
                    raise ValueError(f'plugin {role} path differs from verified candidate')
            participant = next(entry['path'] for entry in candidate['dependencies'] if entry['role'] == 'participant_sample')
            os.environ['LIVE_TEST_PARTICIPANT_SAMPLE'] = participant
            os.environ['LIVE_TEST_PARTICIPANT_TESTS'] = next(entry['path'] for entry in candidate['dependencies'] if entry['role'] == 'participant_tests')
            os.environ['LIVE_TEST_SOURCE_FINGERPRINT'] = source
            os.environ['LIVE_TEST_REFERENCE_IDENTITY'] = json.dumps(dependencies, sort_keys=True)
            if args.writer_plan:
                plan = load_writer_plan(args.writer_plan)
                verify_provenance(plan, source=source, candidate=candidate,
                                  reference=dependencies,
                                  sanitizer=sanitizer_record(os.environ),
                                  fixture=hashlib.sha256(empty_snapshot()).hexdigest())
                os.environ['LIVE_TEST_SEED'] = str(plan['master_seed'])
        if args.writer_case:
            sys.path[:0] = [str(ROOT), str(ROOT / 'tests/reference_service')]
            from tools.live.sanitizer import recorded as sanitizer_record
            from tools.live.test import fingerprint, native_identity
            from tools.live.writer_campaign import validate_case, verify_context
            from writer_history import read_json
            recipe = read_json(args.writer_case, limit=16 * 1024 * 1024)
            writer_case = validate_case(recipe['case'])
            driver = 'plugin' if args.plugin_driver else 'scene'
            if writer_case['driver'] != driver:
                raise ValueError('writer case requires a different driver boundary')
            if driver == 'scene':
                candidate = native_identity(args.scene_transport_driver.resolve(), 'scene_transport')
            verify_context(recipe['context'], source=fingerprint(), candidate=candidate, driver=driver,
                           reference=dependencies,
                           sanitizer=sanitizer_record(os.environ))
            os.environ['LIVE_TEST_SEED'] = str(writer_case['seed'])
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, importlib.metadata.PackageNotFoundError, subprocess.SubprocessError) as error:
        print(f"Reference-service capability unavailable: {error}", file=sys.stderr)
        report_ready(False, f"capability unavailable: {error}")
        return 125
    artifact = Path(os.environ["LIVE_TEST_ARTIFACT_DIR"])
    os.environ["LIVE_TEST_OPENSSL"] = dependencies["openssl"]["path"]
    lane = "kit-reference" if args.kit_executable else "python-reference" if args.python_extension else "plugin-reference" if args.plugin_driver else "resolver-reference" if args.resolver_runtime_driver else "scene-reference" if args.scene_transport_driver else "native-reference" if args.native_driver else "reference"
    generated = artifact / (lane + "-generated")
    generated.mkdir()
    proto_root = ROOT / "source/live/protos"
    schema = proto_root / "nvidia/omniverse/live/v1alpha/live.proto"
    subprocess.run([sys.executable, "-m", "grpc_tools.protoc", f"--proto_path={proto_root}",
                    f"--python_out={generated}", f"--grpc_python_out={generated}", str(schema)],
                   check=True, timeout=30)
    dependencies["proto_sha256"] = hashlib.sha256(schema.read_bytes()).hexdigest()
    dependencies["generated"] = {p.relative_to(generated).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(generated.rglob("*.py"))}
    build_name = lane + "-build.json"
    (artifact / build_name).write_text(json.dumps(dependencies, indent=2, sort_keys=True) + "\n")
    if args.native_driver:
        os.environ["LIVE_TEST_UNARY_DRIVER"] = str(args.native_driver.resolve())
    if args.scene_transport_driver:
        os.environ["LIVE_TEST_SCENE_DRIVER"] = str(args.scene_transport_driver.resolve())
    if args.plugin_driver:
        os.environ["LIVE_TEST_PLUGIN_DRIVER"] = str(args.plugin_driver.resolve())
        os.environ["LIVE_TEST_PLUGIN_PLUG_INFO"] = str(args.plugin_plug_info.resolve())
        os.environ["LIVE_TEST_DIAGNOSTICS_SAMPLE"] = str(args.diagnostics_sample.resolve())
        os.environ.update(PXR_PLUGINPATH_NAME="", PXR_AR_DEFAULT_SEARCH_PATH="", AR_DEFAULT_RESOLVER="")
        if sys.platform == "win32":
            # The driver imports the plugin DLL, which Windows finds on PATH: the package's bin directory, beside
            # its resources directory.
            plugin_bin = args.plugin_plug_info.resolve().parents[1] / "bin"
            os.environ["PATH"] = str(plugin_bin) + os.pathsep + os.environ.get("PATH", "")
        if args.writer_plan:
            os.environ['LIVE_TEST_WRITER_PLAN'] = str(args.writer_plan.resolve())
        else:
            os.environ.pop('LIVE_TEST_WRITER_PLAN', None)
    if args.python_extension:
        extension = args.python_extension.resolve()
        os.environ["LIVE_TEST_PYTHON_EXTENSION"] = str(extension)
        sys.path.insert(0, str(extension.parents[3]))
        from omni.usd.live import _live
        if Path(_live.__file__).resolve() != extension:
            raise RuntimeError("Python imported a different native plugin adapter")
    if args.resolver_runtime_driver:
        os.environ["LIVE_TEST_RESOLVER_RUNTIME_DRIVER"] = str(args.resolver_runtime_driver.resolve())
        os.environ["LIVE_TEST_RESOLVER_RUNTIME_PLUGIN"] = str(args.resolver_runtime_plugin.resolve())
        os.environ["LIVE_TEST_RESOLVER_RUNTIME_PLUG_INFO"] = str(args.resolver_runtime_plug_info.resolve())
    sys.path[:0] = [str(generated), str(ROOT / "tests/reference_service")]
    asyncio.run(serve_one_ping(int(os.environ.get("LIVE_TEST_SEED", "0"))))
    report_ready(True, lane=lane)
    if writer_case is not None:
        from tools.live.writer_campaign import PROFILES, case_passed, soak_scale
        from tools.live.test_support import write_json
        module, cls, method = writer_case['test_id'].rsplit('.', 2)
        case = getattr(importlib.import_module(module), cls)(method)
        case.writer_clients = writer_case['clients']
        case.writer_steps = writer_case['steps']
        case.writer_process_limit = PROFILES[writer_case['profile']]['max_client_processes']
        if writer_case['scenario'] == 'soak':
            case.writer_soak = soak_scale()
        pending = dict(schema_version=1, case=writer_case, passed=False, tests_run=0, peak_clients=0,
                       skipped=0, errors=0, failures=0, unexpected_successes=0, expected_failures=0)
        destination = artifact / 'writer-case-result.json'
        write_json(destination, pending)
        result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([case]))
        pending.update(passed=result.wasSuccessful(), tests_run=result.testsRun,
                       peak_clients=getattr(case, 'writer_peak_clients', 0),
                       workload=getattr(case, 'writer_observed_workload', None),
                       skipped=len(result.skipped), errors=len(result.errors), failures=len(result.failures),
                       unexpected_successes=len(result.unexpectedSuccesses), expected_failures=len(result.expectedFailures))
        write_json(destination, pending)
        return 0 if case_passed(writer_case, pending) else 1
    pattern = "test_kit_process.py" if args.kit_executable else "test_python*.py" if args.python_extension else "test_plugin.py" if args.plugin_driver else "test_resolver.py" if args.resolver_runtime_driver else "test_scene.py" if args.scene_transport_driver else "test_native.py" if args.native_driver else "test_service.py"
    if args.driver_mode:
        # A diagnostic mode is one case of its own; discovery never selects it.
        os.environ["LIVE_TEST_DRIVER_MODE"] = args.driver_mode
        suite = unittest.TestSuite([importlib.import_module("test_native").NativeTests("probe_driver_mode")])
    else:
        suite = (unittest.defaultTestLoader.loadTestsFromName(
            'test_plugin.PluginTests.test_seeded_public_writers_account_every_edit_and_conflict') if args.writer_plan else
            unittest.defaultTestLoader.discover(str(ROOT / "tests/reference_service"), pattern=pattern))
    if suite.countTestCases() == 0:
        print(f"{lane} capability unavailable: no tests discovered", file=sys.stderr)
        return 125
    if args.plugin_driver and not args.writer_plan:
        diagnostics = unittest.defaultTestLoader.discover(str(ROOT / "tests/reference_service"), pattern="test_diagnostics_sample.py")
        if diagnostics.countTestCases() == 0:
            print("Plugin reference capability unavailable: no diagnostics sample tests discovered", file=sys.stderr)
            return 125
        suite.addTests(diagnostics)
        participant = unittest.defaultTestLoader.discover(str(ROOT / "tests/reference_service"), pattern="test_native_sample.py")
        if participant.countTestCases() == 0:
            print("Plugin reference capability unavailable: no participant sample tests discovered", file=sys.stderr)
            return 125
        suite.addTests(participant)
    if args.native_driver and not args.driver_mode:
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / "tests/reference_service"), pattern="test_session.py"))
    graph_pattern = "test_graph_transport.py" if args.scene_transport_driver else "test_graph_service.py" if lane == "reference" else None
    if graph_pattern:
        graph = unittest.defaultTestLoader.discover(str(ROOT / "tests/reference_service"), pattern=graph_pattern)
        if graph.countTestCases() == 0:
            print("Graph reference capability unavailable: no graph tests discovered", file=sys.stderr)
            return 125
        suite.addTests(graph)
    if lane == "reference":
        chained = unittest.defaultTestLoader.discover(str(ROOT / "tests/reference_service"), pattern="test_chained_writes.py")
        if chained.countTestCases() == 0:
            print("Chained-write reference capability unavailable: no chained-write tests discovered", file=sys.stderr)
            return 125
        suite.addTests(chained)
    if args.scene_transport_driver:
        creation = unittest.defaultTestLoader.discover(str(ROOT / "tests/reference_service"), pattern="test_scene_creation.py")
        if creation.countTestCases() == 0:
            print("Scene creation capability unavailable: no creation tests discovered", file=sys.stderr)
            return 125
        suite.addTests(creation)
    available = suite.countTestCases()
    if scenarios or shard:
        try:
            suite = select_cases(suite, scenarios, shard)
        except ValueError as error:
            print(f"{lane} selection invalid: {error}", file=sys.stderr)
            return 125
    discovered = suite.countTestCases()
    if args.kit_executable:
        from tools.live.foundation import test_ids
        kit_tests = sorted(test_ids(suite))
        if kit_tests != sorted(TESTS):
            print("Kit reference capability unavailable: required test inventory changed", file=sys.stderr)
            return 125
    from tools.live.test_support import run_recording
    result, cases = run_recording(suite)
    passed = complete_run(result, discovered)
    if args.kit_executable:
        from tools.live.test import fingerprint
        final_source = fingerprint()
        participant_evidence = {}
        evidence_errors = []
        try:
            inputs_unchanged = snapshot(args) == kit_identity
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError):
            inputs_unchanged = False
            evidence_errors.append('selected_input_validation_failed')
        try:
            participant_evidence = participants(artifact, kit_identity, require_clean=passed)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            evidence_errors.append('participant_evidence_incomplete_or_changed')
        passed = passed and final_source == kit_source and inputs_unchanged and not evidence_errors
        (artifact / 'kit-reference-result.json').write_text(json.dumps(dict(
            schema_version=1, boundary='kit-reference', profile=PROFILE, test_ids=kit_tests, cases=cases,
            passed=passed, discovered=discovered, tests_run=result.testsRun,
            skipped=len(result.skipped), failures=len(result.failures), errors=len(result.errors),
            expected_failures=len(result.expectedFailures), unexpected_successes=len(result.unexpectedSuccesses),
            source_fingerprint=kit_source, final_source_fingerprint=final_source,
            inputs_unchanged=inputs_unchanged, kit_identity=kit_identity, kit_lane=kit_identity['lane'],
            release_package_sha256=release_sha256(kit_identity), participants=participant_evidence,
            evidence_errors=evidence_errors,
            seed=int(os.environ['LIVE_TEST_SEED'])), indent=2) + '\n')
    if not args.kit_executable:
        (artifact / (lane + '-result.json')).write_text(json.dumps(dict(
            schema_version=1, lane=lane, passed=passed, discovered=discovered, tests_run=result.testsRun,
            skipped=len(result.skipped), failures=len(result.failures), errors=len(result.errors),
            expected_failures=len(result.expectedFailures),
            unexpected_successes=len(result.unexpectedSuccesses), cases=cases,
            selection=dict(scenarios=scenarios, shard=None if shard is None else f"{shard[0]}/{shard[1]}",
                           driver_mode=args.driver_mode, available=available)), indent=2) + '\n')
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
