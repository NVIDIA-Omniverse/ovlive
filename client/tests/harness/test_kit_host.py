# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kit lifecycle policy with explicitly completed tasks, not a simulated Kit pass."""
import asyncio
from concurrent.futures import Future
from enum import IntEnum
import importlib.util
import json
import mmap
from pathlib import Path, PureWindowsPath
import struct
import sys
import sysconfig
import tempfile
import threading
import time
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


def maps_identity(identity):
    """A (device, inode) pair as a /proc/self/maps line writes it."""
    return f"{identity[0]} {identity[1]}"


def write_elf(path, needed):
    """Write a minimal 64-bit ELF shared object whose dynamic section lists `needed`."""
    names = b"\0" + b"".join(name.encode() + b"\0" for name in needed)
    offsets, position = [], 1
    for name in needed:
        offsets.append(position)
        position += len(name) + 1
    phoff, phentsize = 64, 56
    dynamic_offset = phoff + 2 * phentsize
    dynamic_size = (len(needed) + 2) * 16
    strtab_offset = dynamic_offset + dynamic_size
    dynamic = b"".join(struct.pack("<qQ", 1, offset) for offset in offsets)
    dynamic += struct.pack("<qQ", 5, strtab_offset) + struct.pack("<qQ", 0, 0)
    total = strtab_offset + len(names)
    header = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9) + struct.pack(
        "<HHIQQQIHHHHHH", 3, 62, 1, 0, phoff, 0, 0, 64, phentsize, 2, 0, 0, 0)
    load = struct.pack("<IIQQQQQQ", 1, 4, 0, 0, 0, total, total, 0x1000)
    dyn = struct.pack("<IIQQQQQQ", 2, 4, dynamic_offset, dynamic_offset, dynamic_offset,
                      dynamic_size, dynamic_size, 8)
    path.write_bytes(header + load + dyn + dynamic + names)


def pe_image(imports, delayed=(), pe32=False, delay_attributes=1):
    """A minimal PE image, laid out as the loader maps it, whose import and delay-import directories list `imports`
    and `delayed`."""
    directories = 96 if pe32 else 112
    optional_size = directories + 16 * 8
    import_rva, delay_rva, names_rva = 0x400, 0x600, 0x800
    names, offsets = b"", []
    for name in (*imports, *delayed):
        offsets.append(names_rva + len(names))
        names += name.encode() + b"\0"
    total = names_rva + len(names)
    image = bytearray(total)
    image[0:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, 0x40)
    image[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", image, 0x44, 0x14C if pe32 else 0x8664, 0, 0, 0, 0, optional_size, 0x2022)
    optional = 0x58
    struct.pack_into("<H", image, optional, 0x10B if pe32 else 0x20B)
    struct.pack_into("<I", image, optional + 56, total)
    struct.pack_into("<I", image, optional + directories - 4, 16)
    if imports:
        struct.pack_into("<II", image, optional + directories + 8 * 1, import_rva, 20 * (len(imports) + 1))
    if delayed:
        struct.pack_into("<II", image, optional + directories + 8 * 13, delay_rva, 32 * (len(delayed) + 1))
    for index, offset in enumerate(offsets[:len(imports)]):
        struct.pack_into("<IIIII", image, import_rva + 20 * index, 0, 0, 0, offset, 0)
    for index, offset in enumerate(offsets[len(imports):]):
        struct.pack_into("<II", image, delay_rva + 32 * index, delay_attributes, offset)
    image[names_rva:] = names
    return bytes(image)


def image_reader(image):
    """A reader of `image` for _pe_imports, which raises OSError past its end as an unreadable page does."""
    def read(rva, size):
        if rva + size > len(image):
            raise OSError("unreadable")
        return image[rva:rva + size]
    return read


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
        # The fixture process loads no USD: model a Linux kit-25.11 host unless a test says otherwise.
        self.bootstrap._verifies_host_usd = lambda: True
        self.bootstrap._host_usd_profile = lambda: "kit-25.11"
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
                # A stock-25.11 package on a kit-25.11 host, an invalid schema version, an unknown profile and an
                # extra key are all rejected before any import.
                for value in (dict(schema_version=1, usd_profile="stock-25.11"),
                              dict(schema_version=True, usd_profile="kit-25.11"),
                              dict(schema_version=1, usd_profile="kit-26.0"),
                              dict(schema_version=1, usd_profile=["kit-25.11"]),
                              dict(schema_version=1, usd_profile="kit-25.11", extra=1)):
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

    def test_host_usd_profile_reads_the_one_loaded_usd_abi(self):
        with tempfile.TemporaryDirectory() as directory:
            maps = Path(directory) / "maps"
            cases = (("7f00-7f01 r-xp 00000000 08:01 1 /opt/kit/lib/libusd_sdf.so\n", "stock-25.11"),
                     ("7f00-7f01 r-xp 00000000 08:01 1 /opt/kit/plugins/libov_25.11usd_ms.so\n", "kit-25.11"),
                     ("7f00-7f01 r-xp 00000000 08:01 1 /usr/lib/libc.so.6\n", None),
                     # Libraries are matched by exact file name, not by a substring of the path.
                     ("7f00-7f01 r-xp 00000000 08:01 1 /opt/kit/lib/libusd_sdf.so.bak\n", None))
            for text, expected in cases:
                maps.write_text(text)
                self.assertEqual(self.probe()._mapped_usd_profile(maps), expected)
            self.assertIsNone(self.probe()._mapped_usd_profile(Path(directory) / "missing"))

            # The ELF reader finds a shim by its DT_NEEDED names, on every platform.
            write_elf(Path(directory) / "shim.so", ["libusd_tf.so", "libusd_sdf.so"])
            self.assertEqual(self.probe()._elf_needed(Path(directory) / "shim.so"), ("libusd_tf.so", "libusd_sdf.so"))
            self.assertEqual(self.probe()._elf_needed(maps), ())
            # A name with no NUL terminator before the end of the file is unverifiable.
            library = Path(directory) / "unterminated.so"
            write_elf(library, ["libusd_tf.so", "libusd_sdf.so"])
            library.write_bytes(library.read_bytes()[:-1])
            self.assertEqual(self.probe()._elf_needed(library), ())
            # A mapped path that is not UTF-8 does not hide the USD mappings.
            maps.write_bytes(b"7f00-7f01 r-xp 00000000 08:01 1 /opt/\xff/libc.so.6\n"
                             b"7f02-7f03 r-xp 00000000 08:01 2 /opt/kit/lib/libusd_sdf.so\n")
            self.assertEqual(self.probe()._mapped_usd_profile(maps), "stock-25.11")
            # A file holding only part of an ELF header is unverifiable, not an error.
            for header in (b"\x7fELF", b"\x7fELF\x02", b"\x7fELF\x02\x01" + bytes(32)):
                (Path(directory) / "truncated").write_bytes(header)
                self.assertEqual(self.probe()._elf_needed(Path(directory) / "truncated"), ())

    def test_host_usd_profile_verifies_every_mapped_shim(self):
        with tempfile.TemporaryDirectory() as directory:
            maps = Path(directory) / "maps"
            stock = "7f00-7f01 r-xp 00000000 08:01 1 /opt/kit/lib/libusd_sdf.so\n"
            # Kit 111 hosts namespaced OVRTX through a USD namespace shim: a libov_25.11usd_ms.so that links the stock
            # libusd_sdf.so forwards to the loaded stock USD, so the host is stock-25.11. A real monolithic library
            # next to stock USD, or a shim that cannot be read, stays ambiguous.
            shim, monolithic = Path(directory) / "shim", Path(directory) / "monolithic"
            (shim / "lib").mkdir(parents=True)
            (monolithic / "lib").mkdir(parents=True)
            write_elf(shim / "lib/libov_25.11usd_ms.so", ["libusd_tf.so", "libusd_sdf.so"])
            write_elf(monolithic / "lib/libov_25.11usd_ms.so", ["libAlembic.so.1.8"])
            shim_library = shim / "lib/libov_25.11usd_ms.so"
            if self.refuses_unverified_shim(maps, stock, shim_library):
                return
            for root, expected in ((shim, "stock-25.11"), (monolithic, None), (Path(directory) / "absent", None)):
                library = root / "lib/libov_25.11usd_ms.so"
                identity = self.mapped(library) if library.exists() else ("08:01", 2)
                maps.write_text(stock + f"7f02-7f03 r-xp 00000000 {maps_identity(identity)} {library}\n")
                self.assertEqual(self.probe()._mapped_usd_profile(maps), expected)
            # Every mapped copy must be verified: a monolithic copy whose file was deleted still occupies the process,
            # so it makes the host ambiguous even beside a readable shim.
            shim_line = f"7f02-7f03 r-xp 00000000 {maps_identity(self.mapped(shim_library))} {shim_library}"
            deleted = "7f04-7f05 r-xp 00000000 08:01 3 /mono/libov_25.11usd_ms.so (deleted)\n"
            maps.write_text(stock + shim_line + "\n" + deleted)
            self.assertIsNone(self.probe()._mapped_usd_profile(maps))
            maps.write_text(stock + shim_line + " (deleted)\n")
            self.assertIsNone(self.probe()._mapped_usd_profile(maps))

    def test_host_usd_check_reads_the_file_that_was_mapped(self):
        with tempfile.TemporaryDirectory() as directory:
            maps = Path(directory) / "maps"
            stock = "7f00-7f01 r-xp 00000000 08:01 1 /opt/kit/lib/libusd_sdf.so\n"
            library = Path(directory) / "lib/libov_25.11usd_ms.so"
            library.parent.mkdir()
            write_elf(library, ["libAlembic.so.1.8"])
            if self.refuses_unverified_shim(maps, stock, library):
                return
            monolithic = self.mapped(library)
            self.assertEqual(self.mapped(library), monolithic)
            # A monolithic library is mapped, and then replaced at its path by a shim after maps is read: the check
            # reads the shim, but the process still holds the monolithic copy.
            replacement = Path(directory) / "replacement.so"
            write_elf(replacement, ["libusd_tf.so", "libusd_sdf.so"])
            replacement.replace(library)
            self.assertNotEqual(self.mapped(library), monolithic)
            self.assertEqual(self.probe()._elf_needed(library, monolithic), ())
            self.assertEqual(self.probe()._elf_needed(library, self.mapped(library)), ("libusd_tf.so", "libusd_sdf.so"))
            maps.write_text(stock + f"7f02-7f03 r-xp 00000000 {maps_identity(monolithic)} {library}\n")
            self.assertIsNone(self.probe()._mapped_usd_profile(maps))
            # One path names two copies when its file was replaced between loads; both must be the shim.
            maps.write_text(stock + f"7f02-7f03 r-xp 00000000 {maps_identity(self.mapped(library))} {library}\n"
                                    f"7f04-7f05 r-xp 00000000 {maps_identity(monolithic)} {library}\n")
            self.assertIsNone(self.probe()._mapped_usd_profile(maps))
            maps.write_text(stock + f"7f02-7f03 r-xp 00000000 {maps_identity(self.mapped(library))} {library}\n")
            self.assertEqual(self.probe()._mapped_usd_profile(maps), "stock-25.11")

    def refuses_unverified_shim(self, maps, stock, library):
        """Off Linux, assert that no shim is ever verified, and return True; on Linux, return False.

        Only Linux names the file behind each mapping, in /proc/self/maps, so only there can the maps reader verify a
        shim. Elsewhere a shim beside stock USD in a maps file stays ambiguous."""
        probe = self.probe()
        if sys.platform.startswith("linux"):
            return False
        write_elf(library, ["libusd_tf.so", "libusd_sdf.so"])
        self.assertEqual(probe._elf_needed(library), ("libusd_tf.so", "libusd_sdf.so"))
        self.assertEqual(probe._elf_needed(library, ("08:01", 2)), ())
        # A maps file names paths with forward slashes, as Linux writes them, whatever the platform's separator.
        shim_line = f"7f02-7f03 r-xp 00000000 08:01 2 {library.as_posix()}\n"
        maps.write_text(shim_line)
        self.assertEqual(probe._mapped_usd_profile(maps), "kit-25.11")
        maps.write_text(stock + shim_line)
        self.assertIsNone(probe._mapped_usd_profile(maps))
        return True

    def test_pe_imports_reads_import_and_delay_import_names(self):
        probe = self.probe()
        image = pe_image(["KERNEL32.dll", "usd_tf.dll"], ["usd_sdf.dll"])
        self.assertEqual(probe._pe_imports(image_reader(image)), ("KERNEL32.dll", "usd_tf.dll", "usd_sdf.dll"))
        self.assertEqual(probe._pe_imports(image_reader(pe_image(["usd_sdf.dll"], pe32=True))), ("usd_sdf.dll",))
        self.assertEqual(probe._pe_imports(image_reader(pe_image([]))), ())
        # An image that cannot be read whole is unverifiable, not an error: a name with no NUL before the end of the
        # image, a delay-import descriptor of virtual addresses, a missing header and an unreadable page.
        self.assertEqual(probe._pe_imports(image_reader(image[:-1])), ())
        self.assertEqual(probe._pe_imports(image_reader(pe_image([], ["usd_sdf.dll"], delay_attributes=0))), ())
        self.assertEqual(probe._pe_imports(image_reader(b"\x7fELF" + image[4:])), ())
        self.assertEqual(probe._pe_imports(image_reader(image[:0x40])), ())
        self.assertEqual(probe._pe_imports(Mock(side_effect=OSError("unreadable"))), ())
        # A directory that runs past SizeOfImage is outside the image, whatever the reader would return there.
        truncated = bytearray(image)
        struct.pack_into("<I", truncated, 0x58 + 56, 0x410)
        self.assertEqual(probe._pe_imports(image_reader(bytes(truncated))), ())

    def test_loaded_usd_profile_reads_the_one_loaded_usd_abi(self):
        probe = self.probe()
        cases = ((["C:\\Kit\\usd\\lib\\usd_sdf.dll"], "stock-25.11"),
                 (["C:\\Kit\\usd\\lib\\USD_SDF.DLL"], "stock-25.11"),
                 (["C:\\Kit\\usd\\lib\\ov_25.11usd_ms.dll"], "kit-25.11"),
                 (["C:\\Windows\\System32\\KERNEL32.DLL"], None),
                 # Modules are matched by exact file name, not by a substring of the path.
                 (["C:\\Kit\\usd\\lib\\usd_sdf.dll.bak", "C:\\usd_sdf.dll\\other.dll"], None))
        for paths, expected in cases:
            with self.subTest(paths=paths):
                modules = [(path, 0x10000 * (index + 1)) for index, path in enumerate(paths)]
                imports = Mock(side_effect=AssertionError("one USD ABI needs no import check"))
                self.assertEqual(probe._loaded_usd_profile(lambda: modules, imports), expected)
        self.assertIsNone(probe._loaded_usd_profile(Mock(side_effect=OSError("cannot list")), Mock()))

    def test_loaded_usd_profile_verifies_every_loaded_shim(self):
        probe = self.probe()
        stock = ("C:\\Kit\\usd\\lib\\usd_sdf.dll", 0x10000)
        images = {0x20000: pe_image(["usd_tf.dll"], ["USD_SDF.dll"]), 0x30000: pe_image(["usd_sdf.dll"]),
                  0x40000: pe_image(["tbb12.dll", "python312.dll"]), 0x50000: pe_image(["usd_sdf.dll"])[:-1]}
        reads = []

        def imports(path, base):
            reads.append(base)
            return probe._pe_imports(image_reader(images[base]))
        shim = "C:\\Kit\\exts\\ovrtx\\bin\\ov_25.11usd_ms.dll"
        # Kit 111's shim imports stock usd_sdf.dll, directly or delay-loaded, and makes the host stock-25.11. Every
        # loaded copy must be the shim: a real monolithic DLL beside it, or a copy whose image cannot be read, is
        # ambiguous.
        cases = (([0x20000], "stock-25.11"), ([0x20000, 0x30000], "stock-25.11"), ([0x40000], None),
                 ([0x20000, 0x40000], None), ([0x50000], None))
        for bases, expected in cases:
            with self.subTest(bases=bases):
                reads.clear()
                modules = [stock] + [(shim, base) for base in bases]
                self.assertEqual(probe._loaded_usd_profile(lambda: modules, imports), expected)
                self.assertEqual(reads, bases[:len(reads)])

    def test_windows_reads_this_process_loaded_modules(self):
        probe = self.probe()
        # Off Windows the loaded-module reader is not used: Linux verifies through its maps file, macOS not at all.
        if sys.platform != "win32":
            self.assertEqual(probe._verifies_host_usd(), sys.platform.startswith("linux"))
            return
        self.assertTrue(probe._verifies_host_usd())
        modules = probe._loaded_modules()
        names = {PureWindowsPath(path).name.lower() for path, _ in modules}
        self.assertIn("kernel32.dll", names)
        # sys.dllhandle is the base address of the Python DLL this process loaded.
        python = [path for path, base in modules if base == sys.dllhandle]
        self.assertEqual(len(python), 1)
        self.assertRegex(PureWindowsPath(python[0]).name.lower(), r"^python3\d*\.dll$")
        self.assertIn("kernel32.dll", [name.lower() for name in probe._module_imports(python[0], sys.dllhandle)])
        # The module read must be the one named at that base: another path, or an address inside the module that is
        # not its base, reads as ().
        self.assertEqual(probe._module_imports("C:\\absent\\other.dll", sys.dllhandle), ())
        self.assertEqual(probe._module_imports(python[0], sys.dllhandle + 0x1000), ())
        # The test process loads no USD.
        self.assertIsNone(probe._host_usd_profile())

    def mapped(self, path):
        """The (device, inode) /proc/self/maps names for a mapping of `path`'s current file."""
        with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY) as data:
            return self.probe()._mapping_identity(data)

    def probe(self):
        """A fresh bootstrap module whose host checks are not modelled."""
        spec = importlib.util.spec_from_file_location("kit_bootstrap_probe",
                                                      ROOT / "source/kit/python/omni/kit/live_services/_bootstrap.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_usd_profile_must_match_the_host_before_python_or_plugin_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "config").mkdir()
            profile = package / "config/live-sdk.json"
            # (package profile, host profile, platform verifies the host, accepted)
            cases = (("stock-25.11", "stock-25.11", True, True),
                     ("kit-25.11", "kit-25.11", True, True),
                     ("kit-25.11", "stock-25.11", True, False),
                     ("stock-25.11", None, True, False),
                     ("kit-25.11", None, True, False),
                     ("kit-25.11", None, False, True),
                     ("stock-25.11", None, False, False))
            for package_profile, host, verifies, accepted in cases:
                with self.subTest(package=package_profile, host=host, verifies=verifies):
                    profile.write_text(json.dumps(dict(schema_version=1, usd_profile=package_profile)))
                    self.bootstrap._verifies_host_usd = lambda: verifies
                    self.bootstrap._host_usd_profile = lambda: host
                    with patch.object(self.bootstrap, "_check_python_abi",
                                      side_effect=RuntimeError("profile accepted")) as python_abi, \
                            patch.object(self.bootstrap.importlib, "import_module") as importer:
                        expected = "profile accepted" if accepted else (
                            "only on Linux and Windows" if not verifies else "but this Kit process loaded")
                        with self.assertRaisesRegex(RuntimeError, expected):
                            self.bootstrap.load_api(package, SimpleNamespace(RegisterPlugins=Mock()))
                        self.assertEqual(python_abi.called, accepted)
                        importer.assert_not_called()

    def test_extension_starts_with_the_schema_extensions(self):
        # OpenUSD fixes its URI-resolver dispatch at the first resolver use, so the `live` resolver must register
        # before order-0 extensions use USD: the slot Kit's schema extensions use.
        text = (ROOT / "source/kit/config/extension.toml").read_text()
        core = text.split("[core]", 1)[1].split("\n[", 1)[0]
        self.assertRegex(core, r"(?m)^order = -100$")

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

    def test_extension_shutdown_gives_an_idle_host_time_to_stop(self):
        module = self.extension_module()
        instance = module.Extension()
        instance.on_startup("idle")
        # A started host stops its threads only with time left: zero answers TIMEOUT.
        self.api._remaining = lambda deadline: 1 if deadline > time.monotonic() else 0
        self.context.close.side_effect = lambda timeout_ms: Result.OK if timeout_ms else Result.TIMEOUT
        with patch.object(module, "close_host", wraps=module.close_host) as close_host:
            instance.on_shutdown()
        close_host.assert_called_once_with(5000)
        self.assertIsNone(module.get_host())
        sys.modules["carb"].log_error.assert_not_called()
        self.runtime.close.assert_called_once()

    def test_extension_shutdown_reports_attached_work_without_waiting(self):
        module = self.extension_module()
        instance = module.Extension()
        instance.on_startup("attached")
        owner = module.get_host()
        self.api._remaining = lambda deadline: 1 if deadline > time.monotonic() else 0
        self.context.close.return_value = Result.BUSY
        instance.on_shutdown()
        self.assertIs(module.get_host(), owner)
        self.runtime.close.assert_not_called()
        message = sys.modules["carb"].log_error.call_args.args[0]
        self.assertIn("BUSY", message)

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
