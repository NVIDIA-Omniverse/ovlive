# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

from tools.live import reference
from tools.live import test as live_runner


@contextlib.contextmanager
def loopback_probe():
    """Mocks the socket class reference.py binds its loopback probe with, through reference.py's own `socket` name.

    Patching the process-wide class instead would break asyncio on Windows, whose event loop builds its self-pipe
    from it."""
    probe = MagicMock()
    with patch.object(reference, "socket", SimpleNamespace(socket=probe)):
        yield probe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))


def plugin_candidate(metadata='metadata', sample='sample'):
    return {'dependencies': [{'role': 'plug_info', 'path': str(Path(metadata).resolve())},
                             {'role': 'diagnostics_sample', 'path': str(Path(sample).resolve())},
                             {'role': 'participant_sample', 'path': str(Path(sample).resolve().with_name('live_participant_sample'))},
                             {'role': 'participant_tests', 'path': str(Path(sample).resolve().with_name('live_participant_tests'))}]}


class Named(unittest.TestCase):
    """A case with a chosen ID that records the readiness file and driver mode it runs under."""
    observed = []

    def __init__(self, identity):
        super().__init__('runTest')
        self.identity = identity

    def id(self):
        return self.identity

    def runTest(self):
        path = os.environ.get('LIVE_TEST_OBSERVE_READY')
        Named.observed.append((self.identity, json.loads(Path(path).read_text()) if path else None))


class FakeNativeTests(unittest.TestCase):
    modes = []

    def probe_driver_mode(self):
        FakeNativeTests.modes.append(os.environ['LIVE_TEST_DRIVER_MODE'])


async def service_answers(seed):
    return None


class ReferenceDispatchTests(unittest.TestCase):
    def setUp(self):
        # The readiness probe needs grpc; the dispatch tests stand in a service that answers.
        ready = patch.object(reference, 'serve_one_ping', new=service_answers)
        ready.start()
        self.addCleanup(ready.stop)
        Named.observed, FakeNativeTests.modes = [], []

    def run_service_lane(self, temporary, arguments=(), environment=None, suites=None):
        """Runs the `reference` lane with trivial discovered cases; returns main()'s status and stderr."""
        ready = Path(temporary) / 'ready.json'
        suites = suites or [unittest.TestSuite([Named('test_service.ServiceTests.test_one'),
                                                Named('test_service.ServiceTests.test_two')]),
                            unittest.TestSuite([Named('test_graph_service.GraphTests.test_one')]),
                            unittest.TestSuite([Named('test_chained_writes.ChainedTests.test_one')])]
        with patch.object(sys, 'argv', ['reference.py', '--openssl', 'openssl', *arguments]), \
                patch.dict(os.environ, dict({'LIVE_TEST_ARTIFACT_DIR': temporary, 'LIVE_TEST_READY_FILE': str(ready),
                                             'LIVE_TEST_OBSERVE_READY': str(ready)}, **(environment or {})), clear=True), \
                patch.object(reference, 'identity', return_value={'openssl': {'path': 'openssl'}}), \
                loopback_probe(), patch.object(reference.subprocess, 'run'), \
                patch.object(reference.unittest.defaultTestLoader, 'discover', side_effect=suites) as discover, \
                patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stderr(io.StringIO()) as errors, \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                status = reference.main()
            except OSError as error:
                status = error
        self.discovered = discover.call_args_list
        return status, errors.getvalue(), json.loads(ready.read_text()) if ready.exists() else None

    def test_readiness_is_reported_once_the_service_answers_and_before_any_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            status, _, record = self.run_service_lane(temporary)
            self.assertEqual(status, 0)
            self.assertEqual((record['ready'], record['lane']), (True, 'reference'))
            self.assertEqual([ready for _, ready in Named.observed], [record] * 4)

    def test_a_service_that_cannot_start_reports_a_failed_start_and_runs_nothing(self):
        async def refuses(seed):
            raise OSError('synthetic bind refusal')
        with tempfile.TemporaryDirectory() as temporary, patch.object(reference, 'serve_one_ping', new=refuses):
            status, _, record = self.run_service_lane(temporary)
        self.assertIsInstance(status, OSError)
        self.assertEqual((record['ready'], record['reason']), (False, 'OSError before the service answered'))
        self.assertEqual((Named.observed, self.discovered), ([], []))

    def test_an_unavailable_capability_is_a_failed_start(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(reference, 'identity', side_effect=RuntimeError('grpcio requires 1.78.0')):
            ready = Path(temporary) / 'ready.json'
            with patch.object(sys, 'argv', ['reference.py', '--openssl', 'openssl']), \
                    patch.dict(os.environ, {'LIVE_TEST_ARTIFACT_DIR': temporary, 'LIVE_TEST_READY_FILE': str(ready)}, clear=True), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(reference.main(), 125)
            record = json.loads(ready.read_text())
        self.assertEqual((record['ready'], record['reason']), (False, 'capability unavailable: grpcio requires 1.78.0'))

    def test_scenarios_and_shards_select_discovered_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            status, _, _ = self.run_service_lane(temporary, environment={
                'LIVE_TEST_SCENARIOS': json.dumps(['test_service.ServiceTests.test_two', 'test_graph_service'])})
            self.assertEqual(status, 0)
            self.assertEqual([identity for identity, _ in Named.observed],
                             ['test_service.ServiceTests.test_two', 'test_graph_service.GraphTests.test_one'])
            evidence = json.loads((Path(temporary) / 'reference-result.json').read_text())
            self.assertEqual((evidence['discovered'], evidence['tests_run'], evidence['selection']['available']), (2, 2, 4))
        with tempfile.TemporaryDirectory() as temporary:
            status, errors, _ = self.run_service_lane(temporary, environment={'LIVE_TEST_SCENARIOS': '["test_missing"]'})
            self.assertEqual(status, 125)
            self.assertIn('scenario selects no case: test_missing', errors)
            self.assertFalse((Path(temporary) / 'reference-result.json').exists())

    def test_a_driver_mode_is_the_only_native_case(self):
        module = Mock(NativeTests=FakeNativeTests)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(reference.importlib, 'import_module', return_value=module) as imported:
            status, _, record = self.run_service_lane(
                temporary, ['--native-driver', 'driver', '--driver-mode', 'actor-auth-timeout'], suites=[])
            self.assertEqual(status, 0)
            imported.assert_called_once_with('test_native')
            evidence = json.loads((Path(temporary) / 'native-reference-result.json').read_text())
        self.assertEqual(FakeNativeTests.modes, ['actor-auth-timeout'])
        self.assertEqual(self.discovered, [])  # Neither test_native.py nor test_session.py is discovered.
        self.assertEqual((evidence['discovered'], evidence['selection']['driver_mode']), (1, 'actor-auth-timeout'))
        self.assertIs(record['ready'], True)
        with patch.object(sys, 'argv', ['reference.py', '--openssl', 'openssl', '--driver-mode', 'actor-auth-timeout']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            reference.main()

    def test_scene_native_and_service_use_distinct_artifacts_and_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifacts"
            artifact.mkdir()
            schema = root / "source/live/protos/nvidia/omniverse/live/v1alpha/live.proto"
            schema.parent.mkdir(parents=True)
            schema.write_text("synthetic proto")
            driver = root / "driver"
            openssl = root / "openssl"
            plugin = root / "plugin.so"
            plug_info = root / "plugInfo.json"
            sample = root / "live_diagnostics_sample"
            cases = (("plugin-reference", ["--plugin-driver", str(driver), "--plugin-plug-info", str(plug_info), "--diagnostics-sample", str(sample)],
                         "test_plugin.py", "LIVE_TEST_PLUGIN_DRIVER"),
                     ("resolver-reference", ["--resolver-runtime-driver", str(driver),
                         "--resolver-runtime-plugin", str(plugin), "--resolver-runtime-plug-info", str(plug_info)],
                         "test_resolver.py", "LIVE_TEST_RESOLVER_RUNTIME_DRIVER"),
                     ("scene-reference", ["--scene-transport-driver", str(driver)], "test_scene.py", "LIVE_TEST_SCENE_DRIVER"),
                     ("native-reference", ["--native-driver", str(driver)], "test_native.py", "LIVE_TEST_UNARY_DRIVER"),
                     ("reference", [], "test_service.py", None))
            for lane, arguments, pattern, variable in cases:
                with self.subTest(lane=lane):
                    def compile_schema(command, **options):
                        self.assertEqual(command[:3], [sys.executable, "-m", "grpc_tools.protoc"])
                        self.assertEqual(command[-1], str(schema))
                        generated = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--python_out=")))
                        self.assertEqual(generated, artifact / (lane + "-generated"))
                        (generated / "live_pb2.py").write_text("synthetic generated code")
                        return subprocess.CompletedProcess(command, 0)

                    outcome = Mock()
                    outcome.wasSuccessful.return_value = True
                    outcome.failures, outcome.errors, outcome.skipped, outcome.unexpectedSuccesses = [], [], [], []
                    outcome.expectedFailures = []
                    outcome.testsRun = 1
                    outcome.cases = [dict(id="test_service.ServiceTests.test_one", outcome="passed", message="")]
                    test_runner = Mock()
                    test_runner.run.return_value = outcome
                    suite = Mock()
                    suite.countTestCases.return_value = 1
                    session_suite = Mock()
                    creation_suite = Mock()
                    with patch.object(reference, "ROOT", root), \
                            patch.object(sys, "argv", ["reference.py", "--openssl", str(openssl), *arguments]), \
                            patch.dict(os.environ, {"LIVE_TEST_ARTIFACT_DIR": str(artifact)}, clear=True), \
                            patch.object(reference, "identity", return_value={"openssl": {"path": str(openssl)}}), \
                            patch.object(live_runner, 'fingerprint', return_value='source'), \
                            patch.object(live_runner, 'native_identity', return_value=plugin_candidate(plug_info, sample)), \
                            loopback_probe() as socket, \
                            patch.object(reference.subprocess, "run", side_effect=compile_schema), \
                            patch.object(reference.unittest.defaultTestLoader, "discover", side_effect=[suite, session_suite, creation_suite]) as discover, \
                            patch.object(reference.unittest, "TextTestRunner", return_value=test_runner), \
                            patch.object(sys, "path", list(sys.path)):
                        self.assertEqual(reference.main(), 0)
                        evidence = json.loads((artifact / (lane + "-result.json")).read_text())
                        self.assertIs(evidence["passed"], True)
                        self.assertEqual(evidence["cases"], outcome.cases)
                        socket.return_value.__enter__.return_value.bind.assert_called_once_with(("127.0.0.1", 0))
                        expected = [call(str(root / "tests/reference_service"), pattern=pattern)]
                        additional = {"native-reference": "test_session.py", "plugin-reference": "test_diagnostics_sample.py",
                                      "scene-reference": "test_graph_transport.py", "reference": "test_graph_service.py"}.get(lane)
                        if additional:
                            expected.append(call(str(root / "tests/reference_service"), pattern=additional))
                            additions = [call(session_suite)]
                            if lane == "plugin-reference":
                                expected.append(call(str(root / "tests/reference_service"), pattern="test_native_sample.py"))
                                additions.append(call(creation_suite))
                            if lane == "scene-reference":
                                expected.append(call(str(root / "tests/reference_service"), pattern="test_scene_creation.py"))
                                additions.append(call(creation_suite))
                            if lane == "reference":
                                expected.append(call(str(root / "tests/reference_service"), pattern="test_chained_writes.py"))
                                additions.append(call(creation_suite))
                            self.assertEqual(suite.addTests.call_args_list, additions)
                        else:
                            suite.addTests.assert_not_called()
                        self.assertEqual(discover.call_args_list, expected)
                        test_runner.run.assert_called_once_with(suite)
                        if variable:
                            self.assertEqual(os.environ[variable], str(driver))
                        if lane == "plugin-reference":
                            self.assertEqual(os.environ["LIVE_TEST_PLUGIN_PLUG_INFO"], str(plug_info))
                            self.assertEqual(os.environ["LIVE_TEST_DIAGNOSTICS_SAMPLE"], str(sample))
                            for key in ("PXR_PLUGINPATH_NAME", "PXR_AR_DEFAULT_SEARCH_PATH", "AR_DEFAULT_RESOLVER"):
                                self.assertEqual(os.environ[key], "")
                        if lane == "resolver-reference":
                            self.assertEqual(os.environ["LIVE_TEST_RESOLVER_RUNTIME_PLUGIN"], str(plugin))
                            self.assertEqual(os.environ["LIVE_TEST_RESOLVER_RUNTIME_PLUG_INFO"], str(plug_info))
                    manifest = json.loads((artifact / (lane + "-build.json")).read_text())
                    self.assertEqual(manifest["proto_sha256"], hashlib.sha256(schema.read_bytes()).hexdigest())
                    self.assertEqual(manifest["generated"], {"live_pb2.py": hashlib.sha256(b"synthetic generated code").hexdigest()})
            # Each lane's build manifest, generated code and completion evidence.
            self.assertEqual(len(list(artifact.iterdir())), 15)

    def test_any_red_fails_the_lane(self):
        red = Mock(**{"id.return_value": "test_scene.SceneTests.test_red"})
        for kind in ("failures", "errors"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                artifact = Path(temporary)
                outcome = Mock()
                outcome.wasSuccessful.return_value = False
                outcome.failures, outcome.errors, outcome.skipped, outcome.unexpectedSuccesses = [], [], [], []
                outcome.expectedFailures = []
                outcome.cases = []
                getattr(outcome, kind).append((red, ""))
                outcome.testsRun = 1
                suite = Mock()
                suite.countTestCases.return_value = 1
                with patch.object(sys, "argv", ["reference.py", "--openssl", "openssl", "--scene-transport-driver", "driver"]), \
                        patch.dict(os.environ, {"LIVE_TEST_ARTIFACT_DIR": temporary}, clear=True), \
                        patch.object(reference, "identity", return_value={"openssl": {"path": "openssl"}}), \
                        loopback_probe(), patch.object(reference.subprocess, "run"), \
                        patch.object(reference.unittest.defaultTestLoader, "discover", side_effect=[suite, Mock(), Mock()]), \
                        patch.object(reference.unittest, "TextTestRunner", return_value=Mock(**{"run.return_value": outcome})), \
                        patch.object(sys, "path", list(sys.path)):
                    self.assertEqual(reference.main(), 1)
                self.assertEqual(sorted(path.name for path in artifact.iterdir()),
                                 ["scene-reference-build.json", "scene-reference-generated", "scene-reference-result.json"])
                evidence = json.loads((artifact / "scene-reference-result.json").read_text())
                self.assertIs(evidence["passed"], False)
                self.assertEqual(evidence[kind], 1)

    def test_every_lane_fails_closed_on_empty_or_skipped_runs(self):
        lanes = (("plugin-reference", ["--plugin-driver", "driver", "--plugin-plug-info", "metadata",
                                       "--diagnostics-sample", "sample"]),
                 ("resolver-reference", ["--resolver-runtime-driver", "driver", "--resolver-runtime-plugin", "plugin",
                                         "--resolver-runtime-plug-info", "metadata"]),
                 ("scene-reference", ["--scene-transport-driver", "driver"]),
                 ("native-reference", ["--native-driver", "driver"]),
                 ("reference", []))
        for (lane, arguments), fault in ((lane, fault) for lane in lanes for fault in ("empty", "skipped")):
            with self.subTest(lane=lane, fault=fault), tempfile.TemporaryDirectory() as temporary:
                artifact = Path(temporary)
                case = unittest.FunctionTestCase(lambda: None)
                outcome = Mock()
                outcome.wasSuccessful.return_value = True
                outcome.failures, outcome.errors, outcome.unexpectedSuccesses, outcome.expectedFailures = [], [], [], []
                outcome.skipped = [(case, "capability absent")]
                outcome.cases = []
                outcome.testsRun = 1
                discovered = (unittest.TestSuite() if fault == "empty" else unittest.TestSuite([case]))
                with patch.object(sys, "argv", ["reference.py", "--openssl", "openssl", *arguments]), \
                        patch.dict(os.environ, {"LIVE_TEST_ARTIFACT_DIR": temporary}, clear=True), \
                        patch.object(reference, "identity", return_value={"openssl": {"path": "openssl"}}), \
                        patch.object(live_runner, 'fingerprint', return_value='source'), \
                        patch.object(live_runner, 'native_identity', return_value=plugin_candidate()), \
                        loopback_probe(), patch.object(reference.subprocess, "run"), \
                        patch.object(reference.unittest.defaultTestLoader, "discover",
                                     side_effect=lambda *_, **__: unittest.TestSuite(list(discovered))), \
                        patch.object(reference.unittest, "TextTestRunner", return_value=Mock(**{"run.side_effect":
                                     lambda ran: (setattr(outcome, "testsRun", ran.countTestCases()), outcome)[1]})) as runner, \
                        patch.object(sys, "path", list(sys.path)), contextlib.redirect_stderr(io.StringIO()) as errors:
                    if fault == "empty":
                        self.assertEqual(reference.main(), 125)
                        self.assertIn(lane + " capability unavailable: no tests discovered", errors.getvalue())
                        runner.assert_not_called()
                        self.assertFalse((artifact / (lane + "-result.json")).exists())
                    else:
                        self.assertEqual(reference.main(), 1)
                        evidence = json.loads((artifact / (lane + "-result.json")).read_text())
                        self.assertIs(evidence["passed"], False)
                        self.assertEqual(evidence["skipped"], 1)

    def test_driver_modes_are_mutually_exclusive_before_capability_probe(self):
        with patch.object(sys, "argv", ["reference.py", "--openssl", "openssl", "--native-driver", "native",
                                        "--scene-transport-driver", "scene"]), \
                patch.object(reference, "identity") as identity, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as rejected:
                reference.main()
            self.assertEqual(rejected.exception.code, 2)
            identity.assert_not_called()

    def test_plugin_requires_driver_and_metadata_as_one_boundary(self):
        for arguments in (["--plugin-driver", "driver"], ["--plugin-plug-info", "metadata"],
                          ["--diagnostics-sample", "sample"],
                          ["--plugin-driver", "driver", "--plugin-plug-info", "metadata"],
                          ["--plugin-driver", "driver", "--native-driver", "native"]):
            with self.subTest(arguments=arguments), \
                    patch.object(sys, "argv", ["reference.py", "--openssl", "openssl", *arguments]), \
                    patch.object(reference, "identity") as identity, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as rejected:
                    reference.main()
                self.assertEqual(rejected.exception.code, 2)
                identity.assert_not_called()

    def test_plugin_empty_discovery_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(sys, "argv", ["reference.py", "--openssl", "openssl",
                                          "--plugin-driver", "driver", "--plugin-plug-info", "metadata",
                                          "--diagnostics-sample", "sample"]), \
                patch.dict(os.environ, {"LIVE_TEST_ARTIFACT_DIR": temporary}, clear=True), \
                patch.object(reference, "identity", return_value={"openssl": {"path": "openssl"}}), \
                patch.object(live_runner, 'fingerprint', return_value='source'), \
                patch.object(live_runner, 'native_identity', return_value=plugin_candidate()), \
                loopback_probe(), \
                patch.object(reference.subprocess, "run"), \
                patch.object(reference.unittest.defaultTestLoader, "discover", return_value=unittest.TestSuite()), \
                patch.object(reference.unittest, "TextTestRunner") as runner, \
                patch.object(sys, "path", list(sys.path)), contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(reference.main(), 125)
            self.assertIn("plugin-reference capability unavailable: no tests discovered", errors.getvalue())
            runner.assert_not_called()

    def test_plugin_missing_diagnostics_cases_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(sys, "argv", ["reference.py", "--openssl", "openssl", "--plugin-driver", "driver",
                                          "--plugin-plug-info", "metadata", "--diagnostics-sample", "sample"]), \
                patch.dict(os.environ, {"LIVE_TEST_ARTIFACT_DIR": temporary}, clear=True), \
                patch.object(reference, "identity", return_value={"openssl": {"path": "openssl"}}), \
                patch.object(live_runner, 'fingerprint', return_value='source'), \
                patch.object(live_runner, 'native_identity', return_value=plugin_candidate()), \
                loopback_probe(), patch.object(reference.subprocess, "run"), \
                patch.object(reference.unittest.defaultTestLoader, "discover", side_effect=[
                    unittest.TestSuite([unittest.FunctionTestCase(lambda: None)]), unittest.TestSuite()]), \
                patch.object(reference.unittest, "TextTestRunner") as runner, \
                patch.object(sys, "path", list(sys.path)), contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(reference.main(), 125)
            self.assertIn("no diagnostics sample tests discovered", errors.getvalue())
            runner.assert_not_called()

    def test_direct_writer_replay_rejects_stale_context_and_selected_paths_before_sockets(self):
        from writer_history import generate
        from scene_wire import empty_snapshot
        for fault in ('source', 'candidate', 'reference', 'sanitizer', 'fixture', 'metadata-path', 'sample-path'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'writer.json'
                candidate = plugin_candidate()
                plan = dict(generate(42, clients=2, steps=4), source_fingerprint='source', candidate=candidate,
                            reference_identity={'openssl': {'path': 'openssl'}}, sanitizer_runtime={},
                            initial_fixture_sha256=hashlib.sha256(empty_snapshot()).hexdigest())
                key = {'source': 'source_fingerprint', 'candidate': 'candidate', 'reference': 'reference_identity',
                       'sanitizer': 'sanitizer_runtime', 'fixture': 'initial_fixture_sha256'}.get(fault)
                if key:
                    plan[key] = 'stale'
                path.write_text(json.dumps(plan))
                arguments = ['reference.py', '--openssl', 'openssl', '--plugin-driver', 'driver',
                             '--plugin-plug-info', 'wrong' if fault == 'metadata-path' else 'metadata',
                             '--diagnostics-sample', 'wrong' if fault == 'sample-path' else 'sample',
                             '--writer-plan', str(path)]
                with patch.object(sys, 'argv', arguments), \
                        patch.dict(os.environ, {'LIVE_TEST_ARTIFACT_DIR': temporary}, clear=True), \
                        patch.object(reference, 'identity', return_value={'openssl': {'path': 'openssl'}}), \
                        patch.object(live_runner, 'fingerprint', return_value='source'), \
                        patch.object(live_runner, 'native_identity', return_value=candidate), \
                        loopback_probe() as socket, \
                        patch.object(reference.subprocess, 'run') as compiler, \
                        patch.object(reference.unittest, 'TextTestRunner') as runner, \
                        patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stderr(io.StringIO()) as errors:
                    self.assertEqual(reference.main(), 125)
                    self.assertIn('differs', errors.getvalue())
                    socket.assert_not_called()
                    compiler.assert_not_called()
                    runner.assert_not_called()

    def test_empty_graph_service_or_transport_suite_cannot_pass(self):
        for arguments in ([], ['--scene-transport-driver', 'driver']):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as temporary, \
                    patch.object(sys, 'argv', ['reference.py', '--openssl', 'openssl', *arguments]), \
                    patch.dict(os.environ, {'LIVE_TEST_ARTIFACT_DIR': temporary}, clear=True), \
                    patch.object(reference, 'identity', return_value={'openssl': {'path': 'openssl'}}), \
                    loopback_probe(), patch.object(reference.subprocess, 'run'), \
                    patch.object(reference.unittest.defaultTestLoader, 'discover', side_effect=[
                        unittest.TestSuite([unittest.FunctionTestCase(lambda: None)]), unittest.TestSuite()]), \
                    patch.object(reference.unittest, 'TextTestRunner') as runner, \
                    patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(reference.main(), 125)
                self.assertIn('no graph tests discovered', errors.getvalue())
                runner.assert_not_called()

    def test_empty_creation_suite_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(sys, 'argv', ['reference.py', '--openssl', 'openssl', '--scene-transport-driver', 'driver']), \
                patch.dict(os.environ, {'LIVE_TEST_ARTIFACT_DIR': temporary}, clear=True), \
                patch.object(reference, 'identity', return_value={'openssl': {'path': 'openssl'}}), \
                loopback_probe(), patch.object(reference.subprocess, 'run'), \
                patch.object(reference.unittest.defaultTestLoader, 'discover', side_effect=[
                    unittest.TestSuite([unittest.FunctionTestCase(lambda: None)]),
                    unittest.TestSuite([unittest.FunctionTestCase(lambda: None)]), unittest.TestSuite()]), \
                patch.object(reference.unittest, 'TextTestRunner') as runner, \
                patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(reference.main(), 125)
            self.assertIn('no creation tests discovered', errors.getvalue())
            runner.assert_not_called()

    def test_resolver_requires_driver_binary_and_metadata_as_one_boundary(self):
        for arguments in (["--resolver-runtime-driver", "driver"],
                          ["--resolver-runtime-plugin", "plugin"],
                          ["--resolver-runtime-driver", "driver", "--resolver-runtime-plugin", "plugin"],
                          ["--native-driver", "native", "--resolver-runtime-driver", "resolver"]):
            with self.subTest(arguments=arguments), \
                    patch.object(sys, "argv", ["reference.py", "--openssl", "openssl", *arguments]), \
                    patch.object(reference, "identity") as identity, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as rejected:
                    reference.main()
                self.assertEqual(rejected.exception.code, 2)
                identity.assert_not_called()


if __name__ == "__main__":
    unittest.main()
