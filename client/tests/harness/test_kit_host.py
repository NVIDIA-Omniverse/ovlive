# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kit lifecycle policy with explicitly completed tasks, not a simulated Kit pass."""
import asyncio
from concurrent.futures import Future
from enum import IntEnum
import importlib.util
import json
from pathlib import Path
import sys
import sysconfig
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


class Result(IntEnum):
    OK = 0
    NOT_INITIALIZED = 3
    TIMEOUT = 9
    CANCELLED = 10
    RESOURCE_EXHAUSTED = 13
    BUSY = 14
    INTERNAL_ERROR = 20


class LiveError(RuntimeError):
    def __init__(self, result):
        self.result = result


class KitHostTests(unittest.TestCase):
    def setUp(self):
        self.context = SimpleNamespace(close=Mock(return_value=Result.OK), _failed=Mock())
        self.runtime = SimpleNamespace(close=Mock(return_value=Result.OK), register_endpoint=Mock(return_value=object()))
        self.api = ModuleType("kit_host_fixture")
        self.api.Result, self.api.LiveError = Result, LiveError
        self.api._owners_lock = threading.RLock()
        self.api.initialize = Mock(return_value=self.runtime)
        self.api.set_publication_context = Mock(return_value=self.context)
        self.api.register_auth_required_callback = Mock(return_value=object())
        self.api.register_status_callback = Mock(return_value=object())
        self.api._timeout = lambda value: value
        self.api._remaining = lambda deadline: 0
        name = self.api.__name__ + "._kit_host"
        spec = importlib.util.spec_from_file_location(name, ROOT / "source/bindings/python/omni/usd/live/_kit_host.py")
        self.module = importlib.util.module_from_spec(spec)
        modules = patch.dict(sys.modules, {self.api.__name__: self.api, name: self.module})
        modules.start()
        self.addCleanup(modules.stop)
        spec.loader.exec_module(self.module)
        spec = importlib.util.spec_from_file_location("kit_bootstrap_fixture", ROOT / "source/kit/python/omni/kit/live_services/_bootstrap.py")
        self.bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bootstrap)
        self.tasks = []
        def submit(coroutine):
            future = Future()
            self.tasks.append((coroutine, future))
            return future
        self.host = self.module.KitHost(submit, max_endpoints=2)
        self.addCleanup(self.retire_tasks)

    def retire_tasks(self):
        for coroutine, future in self.tasks:
            coroutine.close()
            if not future.done():
                future.cancel()

    @staticmethod
    def extension(package, suffix=None):
        """Name the package's native extension; its bytes are never loaded."""
        path = package / "python/omni/usd/live" / ("_live" + (suffix or sysconfig.get_config_var("EXT_SUFFIX")))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"never loaded")

    def endpoint(self):
        return self.host.register_endpoint("localhost:43123", on_auth_required=lambda event: None,
                                           on_status=lambda event: None, access_token="synthetic")

    def test_busy_context_preserves_auth_and_runtime_until_usd_retires(self):
        self.host.start()
        owned = self.endpoint()
        self.context.close.side_effect = [Result.BUSY, Result.OK, Result.OK]
        self.runtime.close.side_effect = [Result.TIMEOUT, Result.OK]
        self.assertEqual(self.host.close(), Result.BUSY)
        self.runtime.close.assert_not_called()
        self.assertIs(self.host.endpoints[0], owned)
        with self.assertRaises(LiveError) as error:
            self.endpoint()
        self.assertEqual(error.exception.result, Result.NOT_INITIALIZED)
        self.assertEqual(self.host.close(), Result.TIMEOUT)
        self.assertIs(self.host.endpoints[0], owned)
        self.assertEqual(self.host.close(), Result.OK)
        self.assertEqual(self.host.endpoints, ())
        self.assertEqual(self.host.close(), Result.OK)
        self.assertEqual(self.runtime.close.call_count, 2)

    def test_partial_startup_and_endpoint_setup_remain_owned(self):
        self.api.set_publication_context.side_effect = RuntimeError("fixture registration failure")
        with self.assertRaises(RuntimeError):
            self.host.start()
        self.assertIs(self.host.runtime, self.runtime)
        self.assertEqual(self.host.close(), Result.OK)
        self.runtime.close.assert_called_once()
        self.api.set_publication_context.side_effect = None
        host = self.module.KitHost(lambda coroutine: None)
        host.start()
        self.api.register_status_callback.side_effect = RuntimeError("fixture observer failure")
        with self.assertRaises(RuntimeError):
            host.register_endpoint("localhost:43123", on_auth_required=lambda event: None, on_status=lambda event: None)
        self.assertIs(host.endpoints[0].endpoint, self.runtime.register_endpoint.return_value)
        self.assertIs(host.endpoints[0].auth, self.api.register_auth_required_callback.return_value)
        self.assertIsNone(host.endpoints[0].status)

    def test_failed_registration_does_not_consume_capacity_and_successes_are_bounded(self):
        self.host.start()
        self.runtime.register_endpoint.side_effect = LiveError(Result.RESOURCE_EXHAUSTED)
        for _ in range(3):
            with self.assertRaises(LiveError):
                self.endpoint()
        self.assertEqual(self.host.endpoints, ())
        self.runtime.register_endpoint.side_effect = None
        self.endpoint(); self.endpoint()
        calls = self.runtime.register_endpoint.call_count
        with self.assertRaises(LiveError) as error:
            self.endpoint()
        self.assertEqual(error.exception.result, Result.RESOURCE_EXHAUSTED)
        self.assertEqual(self.runtime.register_endpoint.call_count, calls)

    def test_dispatch_runs_only_on_explicit_completion_and_cancellation_disarms(self):
        self.host.start()
        job = Mock()
        self.host._dispatch(job)
        job.assert_not_called()
        coroutine, future = self.tasks[0]
        asyncio.run(coroutine)
        future.set_result(None)
        job.assert_called_once_with()
        self.context._failed.assert_not_called()
        cancelled = Mock()
        self.host._dispatch(cancelled)
        self.tasks[1][1].cancel()
        cancelled.assert_not_called()
        cancelled.cancel.assert_called_once_with()
        self.context._failed.assert_called_once_with(Result.CANCELLED)

    def test_job_exception_becomes_typed_failure_and_off_host_setup_rejects(self):
        self.host.start()
        job = Mock(side_effect=LiveError(Result.BUSY))
        self.host._dispatch(job)
        coroutine, future = self.tasks[0]
        try:
            asyncio.run(coroutine)
        except LiveError as error:
            future.set_exception(error)
        self.context._failed.assert_called_once_with(Result.BUSY)
        job.cancel.assert_called_once_with()
        observed = []
        def wrong_thread():
            try:
                self.host.close()
            except LiveError as error:
                observed.append(error.result)
        worker = threading.Thread(target=wrong_thread)
        worker.start(); worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(observed, [Result.BUSY])
        self.runtime.close.assert_not_called()

    def test_plugin_collision_rejects_before_registry_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "resources").mkdir()
            library = package / self.bootstrap._LIBRARY
            library.parent.mkdir()
            library.write_bytes(b"synthetic, never loaded")
            (package / "resources/plugInfo.json").write_text("{}")
            ours = SimpleNamespace(name="omni_usd_live", path=str(library), metadata={"Types": {"format": {"extensions": ["live"]}}})
            registry = SimpleNamespace(GetAllPlugins=Mock(return_value=[ours]), RegisterPlugins=Mock(),
                                       GetPluginWithName=Mock(return_value=ours))
            self.assertIs(self.bootstrap.register_plugin(package, registry), ours)
            for name, types in (("competing", {"format": {"extensions": ["live"]}}),
                                ("other", {"resolver": {"uriSchemes": ["live"]}}),
                                ("omni_usd_live", {})):
                registry.GetAllPlugins.return_value = [SimpleNamespace(name=name, path="/different/plugin.so", metadata={"Types": types})]
                registry.RegisterPlugins.reset_mock()
                with self.assertRaises(RuntimeError):
                    self.bootstrap.register_plugin(package, registry)
                registry.RegisterPlugins.assert_not_called()

    def test_profile_and_late_plugin_conflicts_reject_before_native_import(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "config").mkdir(); (package / "resources").mkdir()
            (package / self.bootstrap._LIBRARY).parent.mkdir()
            (package / self.bootstrap._LIBRARY).write_bytes(b"never loaded")
            (package / "resources/plugInfo.json").write_text("{}")
            profile = package / "config/live-sdk.json"
            registry = SimpleNamespace(GetAllPlugins=Mock(return_value=[]), RegisterPlugins=Mock())
            with patch.object(self.bootstrap.importlib, "import_module") as importer:
                for value in (dict(schema_version=1, usd_profile="stock-25.11"),
                              dict(schema_version=True, usd_profile="kit-25.11")):
                    profile.write_text(json.dumps(value))
                    with self.assertRaisesRegex(RuntimeError, "kit-25.11"):
                        self.bootstrap.load_api(package, registry)
                importer.assert_not_called()
                registry.GetAllPlugins.assert_not_called()
            profile.write_text(json.dumps(dict(schema_version=1, usd_profile="kit-25.11")))
            self.extension(package)
            def import_parent(name):
                self.assertEqual(name, "omni.usd")
                registry.GetAllPlugins.return_value = [SimpleNamespace(name="competing", path="/unused",
                    metadata={"Types": {"format": {"extensions": ["live"]}}})]
                return object()
            with patch.object(self.bootstrap.importlib, "import_module", side_effect=import_parent) as importer:
                with self.assertRaisesRegex(RuntimeError, "competing"):
                    self.bootstrap.load_api(package, registry)
                importer.assert_called_once_with("omni.usd")
                registry.RegisterPlugins.assert_not_called()

    def test_conflicting_python_api_rejects_before_registry_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "config").mkdir()
            (package / "config/live-sdk.json").write_text(json.dumps(dict(schema_version=1, usd_profile="kit-25.11")))
            self.extension(package)
            parent = SimpleNamespace(__path__=["existing-host-package"], __file__="/kit/exts/omni.usd/omni/usd/__init__.py")
            other = SimpleNamespace(__file__="/different/omni/usd/live/__init__.py")
            registry = SimpleNamespace(RegisterPlugins=Mock())
            for loaded in (True, False):
                with self.subTest(loaded=loaded), patch.object(self.bootstrap, "_check_plugins"), \
                        patch.dict(sys.modules, {"omni.usd.live": other if loaded else None}), \
                        patch.object(self.bootstrap.importlib, "import_module", return_value=parent) as importer, \
                        patch.object(self.bootstrap.importlib.util, "find_spec", return_value=SimpleNamespace(origin=other.__file__)):
                    with self.assertRaisesRegex(RuntimeError, "different Live Services Python package"):
                        self.bootstrap.load_api(package, registry)
                    registry.RegisterPlugins.assert_not_called()
                    importer.assert_called_once_with("omni.usd")
                    self.assertEqual(parent.__path__, ["existing-host-package"])

    def test_regular_usd_parent_keeps_its_paths_and_imports_only_canonical_child(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "config").mkdir()
            (package / "config/live-sdk.json").write_text(json.dumps(dict(schema_version=1, usd_profile="kit-25.11")))
            self.extension(package)
            parent = ModuleType("omni.usd")
            parent.__file__ = "/kit/exts/omni.usd/omni/usd/__init__.py"
            paths = parent.__path__ = ["existing-host-package"]
            api = ModuleType("omni.usd.live")
            api.__file__ = str(package / "python/omni/usd/live/__init__.py")
            with patch.object(self.bootstrap, "_check_plugins"), patch.object(self.bootstrap, "register_plugin"), \
                    patch.object(self.bootstrap.importlib, "import_module", side_effect=[parent, api]) as importer, \
                    patch.object(self.bootstrap.importlib.util, "find_spec", return_value=None):
                self.assertIs(self.bootstrap.load_api(package, object()), api)
                self.assertIs(parent.__path__, paths)
                self.assertEqual(paths, ["existing-host-package", str(package.resolve() / "python/omni/usd")])
                self.assertEqual([call.args for call in importer.call_args_list], [("omni.usd",), ("omni.usd.live",)])

    def test_namespace_usd_parent_rejects_before_plugin_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "config").mkdir()
            (package / "config/live-sdk.json").write_text(json.dumps(dict(schema_version=1, usd_profile="kit-25.11")))
            self.extension(package)
            parent = ModuleType("omni.usd")
            parent.__path__ = [str(package / "python/omni/usd")]
            parent.__file__ = None  # What the import system sets for a namespace package.
            with patch.object(self.bootstrap, "_check_plugins"), \
                    patch.object(self.bootstrap, "register_plugin") as register, \
                    patch.object(self.bootstrap.importlib, "import_module", return_value=parent) as importer:
                with self.assertRaisesRegex(RuntimeError, "namespace package"):
                    self.bootstrap.load_api(package, object())
                importer.assert_called_once_with("omni.usd")
                register.assert_not_called()
            self.assertEqual(parent.__path__, [str(package / "python/omni/usd")])

    def test_python_abi_mismatch_rejects_before_registry_or_import(self):
        for suffixes in ((), (".cpython-299-x86_64-linux-gnu.so",),
                         (sysconfig.get_config_var("EXT_SUFFIX"), ".cpython-299-x86_64-linux-gnu.so")):
            with self.subTest(suffixes=suffixes), tempfile.TemporaryDirectory() as directory:
                package = Path(directory)
                (package / "config").mkdir()
                (package / "config/live-sdk.json").write_text(json.dumps(dict(schema_version=1, usd_profile="kit-25.11")))
                for suffix in suffixes:
                    self.extension(package, suffix)
                registry = SimpleNamespace(GetAllPlugins=Mock(return_value=[]), RegisterPlugins=Mock())
                with patch.object(self.bootstrap.importlib, "import_module") as importer:
                    with self.assertRaisesRegex(RuntimeError, "Python extension"):
                        self.bootstrap.load_api(package, registry)
                importer.assert_not_called()
                registry.GetAllPlugins.assert_not_called()
                registry.RegisterPlugins.assert_not_called()

    def test_extension_depends_on_kit_omni_usd(self):
        text = (ROOT / "source/kit/config/extension.toml").read_text()
        dependencies = text.split("[dependencies]", 1)[1].split("\n[", 1)[0]
        self.assertRegex(dependencies, r'(?m)^"omni\.usd" = \{\}$')

    @staticmethod
    def declared_python_modules():
        """The `[[python.module]]` tables of the extension configuration, in declaration order."""
        text = (ROOT / "source/kit/config/extension.toml").read_text()
        tables = []
        for table in text.split("[[python.module]]\n")[1:]:
            entry = {}
            for line in table.split("\n[", 1)[0].splitlines():
                if line.strip() and not line.startswith("#"):
                    key, value = (part.strip() for part in line.split("=", 1))
                    entry[key] = json.loads(value)
            tables.append(entry)
        return tables

    def test_extension_declares_bootstrap_module_before_public_api_module(self):
        # Kit's API checker treats exactly the declared module names as public; the bootstrap module comes first.
        self.assertEqual(self.declared_python_modules(), [
            {"name": "omni.kit.live_services", "path": "python"},
            {"name": "omni.usd.live", "path": "python", "searchExt": False},
        ])

    def kit_startup(self, extension, events):
        """Kit's first enable of the declared modules, as omni.ext's ExtensionModules.startup performs it.

        Each module, in declaration order, is taken from sys.modules or imported, and the IExt subclasses of a module
        whose entry searches for them start before the next module is imported; an exception ends startup.
        """
        iext, started, modules = sys.modules["omni.ext"].IExt, [], []

        def import_module(name):
            events.append(("import", name))
            if name != "omni.kit.live_services":
                raise AssertionError("Kit imported " + name + " before the bootstrap loaded it")
            return extension

        self.addCleanup(lambda: [instance.on_shutdown() for instance in reversed(started)])
        for entry in self.declared_python_modules():
            module = sys.modules.get(entry["name"]) or import_module(entry["name"])
            modules.append(module)
            if entry.get("searchExt", True):
                for name in dir(module):
                    candidate = getattr(module, name)
                    if isinstance(candidate, type) and issubclass(candidate, iext):
                        instance = candidate()
                        started.append(instance)
                        instance.on_startup("omni_usd_live")
        return modules

    def test_kit_startup_reaches_public_api_module_only_after_bootstrap_checks(self):
        extension = self.extension_module(api_loaded=False)
        events = []

        def load_api(package, registry):
            events.append(("load_api",))
            sys.modules["omni.usd.live"] = self.api
            return self.api

        extension.load_api = Mock(side_effect=load_api)
        self.assertEqual(self.kit_startup(extension, events), [extension, self.api])
        self.assertEqual(events, [("import", "omni.kit.live_services"), ("load_api",)])
        self.assertIsNotNone(extension.get_host())

    def test_failed_bootstrap_check_ends_kit_startup_before_public_api_module(self):
        extension = self.extension_module(api_loaded=False)
        events = []

        def load_api(package, registry):
            events.append(("load_api",))
            raise RuntimeError("Kit requires the explicit kit-25.11 Live Services package")

        extension.load_api = Mock(side_effect=load_api)
        with self.assertRaisesRegex(RuntimeError, "kit-25.11"):
            self.kit_startup(extension, events)
        self.assertEqual(events, [("import", "omni.kit.live_services"), ("load_api",)])
        self.assertNotIn("omni.usd.live", sys.modules)

    def test_kit_restart_of_extension_in_one_process_fails_before_public_api_module(self):
        extension = self.extension_module(api_loaded=False)
        events = []

        def load_api(package, registry):
            events.append(("load_api",))
            sys.modules["omni.usd.live"] = self.api
            return self.api

        extension.load_api = Mock(side_effect=load_api)
        self.kit_startup(extension, events)
        owner = extension.get_host()
        self.assertIsNotNone(owner)
        self.assertEqual(extension.close_host(), Result.OK)
        # Kit's shutdown removes the extension's modules; a reload of omni.usd.live would run only after on_startup.
        del sys.modules["omni.usd.live"]
        events.clear()
        with self.assertRaisesRegex(RuntimeError, "once per Kit process; restart Kit"):
            self.kit_startup(extension, events)
        self.assertEqual(events, [("import", "omni.kit.live_services")])
        self.assertNotIn("omni.usd.live", sys.modules)
        self.assertEqual(extension.load_api.call_count, 1)
        self.runtime.close.assert_called_once()

    def extension_module(self, api_loaded=True):
        omni = ModuleType("omni")
        omni.__path__ = []
        omni.ext = ModuleType("omni.ext")
        omni.ext.IExt = type("IExt", (), {})
        engine = ModuleType("omni.kit.async_engine")
        engine.run_coroutine = Mock()
        carb = ModuleType("carb")
        carb.log_error = Mock()
        pxr = ModuleType("pxr")
        pxr.Plug = SimpleNamespace(Registry=Mock())
        name = "kit_extension_fixture"
        self.loop = asyncio.new_event_loop()
        self.addCleanup(self.loop.close)
        loop_patch = patch.object(asyncio, "get_event_loop", return_value=self.loop)
        loop_patch.start()
        self.addCleanup(loop_patch.stop)
        spec = importlib.util.spec_from_file_location(name, ROOT / "source/kit/python/omni/kit/live_services/__init__.py")
        extension = importlib.util.module_from_spec(spec)
        modules = patch.dict(sys.modules, {name: extension, name + "._bootstrap": self.bootstrap,
            "omni": omni, "omni.ext": omni.ext, "omni.kit.async_engine": engine, "carb": carb, "pxr": pxr,
            "omni.usd.live": self.api, "omni.usd.live._kit_host": self.module})
        modules.start(); self.addCleanup(modules.stop)
        if not api_loaded:
            del sys.modules["omni.usd.live"]
        spec.loader.exec_module(extension)
        extension.load_api = Mock(return_value=self.api)
        # Each test is a fresh Kit process: no earlier start has marked sys.
        marker = patch.object(sys, extension._STARTED, False, create=True)
        marker.start(); self.addCleanup(marker.stop)
        return extension

    def test_failed_extension_instance_never_closes_another_instances_owner(self):
        module = self.extension_module()
        first, rejected = module.Extension(), module.Extension()
        first.on_startup("first")
        owner = module.get_host()
        with self.assertRaisesRegex(RuntimeError, "restart Kit"):
            rejected.on_startup("second")
        rejected.on_shutdown()
        self.assertIs(module.get_host(), owner)
        self.runtime.close.assert_not_called()
        first.on_shutdown()
        self.assertIsNone(module.get_host())
        self.runtime.close.assert_called_once()
        replacement = module.Extension()
        with self.assertRaisesRegex(RuntimeError, "restart Kit"):
            replacement.on_startup("replacement")
        first.on_shutdown()
        replacement.on_shutdown()
        self.assertIsNone(module.get_host())
        self.assertEqual(self.runtime.close.call_count, 1)
        self.assertEqual(module.load_api.call_count, 1)

    def test_extension_submits_publication_jobs_thread_safely_to_the_host_loop(self):
        module = self.extension_module()
        instance = module.Extension()
        instance.on_startup("thread-safe")
        owner = module.get_host()
        ran, submitted = [], []

        async def job():
            ran.append(threading.get_ident())

        # The publication consumer thread calls _dispatch; submission must not run
        # the job there or touch the loop unsafely.
        worker = threading.Thread(target=lambda: submitted.append(owner._submit(job())))
        worker.start()
        worker.join()
        self.assertEqual(ran, [])
        self.loop.run_until_complete(asyncio.wrap_future(submitted[0], loop=self.loop))
        self.assertEqual(ran, [threading.get_ident()])
        sys.modules["omni.kit.async_engine"].run_coroutine.assert_not_called()
        instance.on_shutdown()
        self.assertIsNone(module.get_host())

    def test_extension_retains_partial_startup_for_its_own_cleanup(self):
        module = self.extension_module()
        instance = module.Extension()
        self.api.set_publication_context.side_effect = RuntimeError("controlled startup failure")
        with self.assertRaises(RuntimeError):
            instance.on_startup("partial")
        self.assertIs(module.get_host(), instance._owner)
        instance.on_shutdown()
        self.assertIsNone(module.get_host())
        self.runtime.close.assert_called_once()

    def test_session_catalogue_ownership_blocks_runtime_close(self):
        self.host.start()
        registration = self.host.register_endpoint("host", on_auth_required=lambda e: None, on_status=lambda e: None)
        catalog = SimpleNamespace(closed=False, close=Mock(return_value=Result.BUSY))
        factory = Mock(return_value=catalog)
        module = SimpleNamespace(SessionCatalog=factory)
        with patch.dict(sys.modules, {self.api.__name__ + "._kit_sessions": module}):
            with self.assertRaises(ValueError):
                self.host.session_catalog(SimpleNamespace(endpoint=registration.endpoint), "base", current_user="alice")
            self.assertIs(self.host.session_catalog(registration, "base", current_user="alice"), catalog)
        factory.assert_called_once_with(registration.endpoint, "base", current_user="alice")
        self.assertEqual(self.host.close(), Result.BUSY)
        self.context.close.assert_called_once()
        self.runtime.close.assert_not_called()
        self.assertEqual(self.host.session_catalogs, (catalog,))
        catalog.close.return_value = Result.OK
        self.assertEqual(self.host.close(), Result.OK)
        self.runtime.close.assert_called_once()

    def test_status_observer_is_bound_to_this_host_runtime(self):
        self.host.start()
        self.runtime._subscribe = Mock(return_value=object())
        callback, on_error = Mock(), Mock()
        result = self.host.observe_status(callback, on_error=on_error)
        self.assertIs(result, self.runtime._subscribe.return_value)
        self.runtime._subscribe.assert_called_once_with("status", None, callback, on_error)
        self.host.close()
        with self.assertRaises(LiveError):
            self.host.observe_status(callback)
