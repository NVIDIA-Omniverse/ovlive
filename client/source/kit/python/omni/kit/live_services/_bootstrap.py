# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the package and USD registry before importing native Live Services."""
import ctypes
import importlib
import importlib.util
import json
import mmap
from pathlib import Path, PureWindowsPath
import struct
import sys
import sysconfig


# The plugin library within the package: beside the samples on Windows, where the loader finds an imported DLL, and
# under lib/ elsewhere.
_LIBRARY = ("bin/omni_usd_live.dll" if sys.platform == "win32" else
            "lib/libomni_usd_live.dylib" if sys.platform == "darwin" else "lib/libomni_usd_live.so")


# A Live Services package binds to exactly one USD ABI. Kit hosts load either the monolithic ov-namespace build
# (kit-25.11) or, from Kit 111, the stock pxr split build (stock-25.11). On Linux and Windows the bootstrap reads which
# one this process loaded, and accepts only the package that matches it.
_HOST_USD_MARKERS = {"kit-25.11": "/libov_25.11usd_ms.so", "stock-25.11": "/libusd_sdf.so"}
# The same marker libraries as Windows names them; module file names compare case-insensitively.
_HOST_USD_MODULES = {"kit-25.11": "ov_25.11usd_ms.dll", "stock-25.11": "usd_sdf.dll"}


def _verifies_host_usd():
    """Whether this platform can read the host's loaded USD: Linux from /proc/self/maps, Windows from its modules."""
    return sys.platform.startswith("linux") or sys.platform == "win32"


def _host_usd_profile():
    """The Live Services USD profile this process has loaded, or None when there is none or it is ambiguous."""
    return _loaded_usd_profile() if sys.platform == "win32" else _mapped_usd_profile()


def _mapping_identity(data, maps_path="/proc/self/maps"):
    """The (device, inode) the kernel reports for the file behind the mapping `data`, an mmap.ACCESS_COPY object.

    The kernel names a mapped file by the identity it reports in /proc/self/maps, which on overlayfs differs from what
    stat() returns for the same file; reading it from a mapping of our own compares like with like."""
    anchor = ctypes.c_char.from_buffer(data)
    start = ctypes.addressof(anchor)
    del anchor
    return next((identity for address, identity, _, _ in _mappings(maps_path) if address == start), None)


def _elf_needed(path, identity=None):
    """DT_NEEDED names of a 64-bit little-endian ELF shared object; () if it cannot be read.

    With `identity`, the (device, inode) of a mapping in this process, the names are read only from that same file:
    one that has since been replaced at `path` reads as ()."""
    try:
        with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY) as data:
            if identity is not None and _mapping_identity(data) != identity:
                return ()
            # An ELF64 header is 64 bytes; the fields read below lie within it.
            if len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
                return ()
            (phoff,) = struct.unpack_from("<Q", data, 0x20)
            phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
            loads, dynamic = [], None
            for index in range(phnum):
                p_type, _, p_offset, p_vaddr, _, p_filesz, _, _ = struct.unpack_from(
                    "<IIQQQQQQ", data, phoff + index * phentsize)
                if p_type == 1:
                    loads.append((p_vaddr, p_offset, p_filesz))
                elif p_type == 2:
                    dynamic = (p_offset, p_filesz)
            if dynamic is None:
                return ()
            needed, strtab = [], None
            for offset in range(dynamic[0], dynamic[0] + dynamic[1] - 15, 16):
                tag, value = struct.unpack_from("<qQ", data, offset)
                if tag == 0:
                    break
                if tag == 1:
                    needed.append(value)
                elif tag == 5:
                    strtab = value
            base = next((o + strtab - v for v, o, size in loads
                         if strtab is not None and v <= strtab < v + size), None)
            if base is None:
                return ()
            names = []
            for n in needed:
                # A name with no NUL before the end of the file is malformed: the file is unverifiable.
                end = data.find(b"\0", base + n)
                if end == -1:
                    return ()
                names.append(data[base + n:end].decode())
            return tuple(names)
    except (OSError, ValueError, struct.error, UnicodeDecodeError):
        return ()


def _mappings(maps_path):
    """Each file mapping in a maps file: (start address, (device, inode), path, whether the file was deleted)."""
    # A mapped path can hold bytes that are not UTF-8; they decode as surrogate escapes, which open() maps back.
    for line in Path(maps_path).read_bytes().decode(errors="surrogateescape").splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6:
            continue
        try:
            start = int(fields[0].split("-", 1)[0], 16)
            inode = int(fields[4])
        except ValueError:
            continue
        # A mapping whose file was deleted keeps its old path with a " (deleted)" suffix, and a file at that path, if
        # any, is not the one mapped.
        path = fields[5]
        deleted = path.endswith(" (deleted)")
        if deleted:
            path = path[:-len(" (deleted)")]
        yield start, (fields[3], inode), path, deleted


def _mapped_usd_profile(maps_path="/proc/self/maps"):
    """The Live Services USD profile a Linux maps file shows loaded, or None when there is none or it is ambiguous."""
    try:
        mappings = list(_mappings(maps_path))
    except OSError:
        return None
    # Every mapped copy of a marker library, by its exact file name and the (device, inode) of the file mapped: one
    # path can name more than one copy when its file was replaced between loads.
    mapped = {profile: set() for profile in _HOST_USD_MARKERS}
    for _, identity, path, deleted in mappings:
        for profile, library in _HOST_USD_MARKERS.items():
            if path.endswith(library):
                mapped[profile].add((path, identity, deleted))
    found = [profile for profile, copies in mapped.items() if copies]
    if len(found) == 1:
        return found[0]
    if len(found) == 2:
        # Kit 111 hosts namespaced OVRTX and OVStage through a USD namespace shim: a library named
        # libov_25.11usd_ms.so that forwards to the stock split USD this process already loaded. Only that
        # forwarder, which links libusd_sdf.so, leaves one USD ABI in the process. Every mapped copy must be verified
        # from the file it was mapped from: a real monolithic library next to stock USD, one that was deleted, one
        # replaced at its path since it was mapped, or one that cannot be read is ambiguous.
        if all(not deleted and "libusd_sdf.so" in _elf_needed(path, identity)
               for path, identity, deleted in mapped["kit-25.11"]):
            return "stock-25.11"
    return None


# A PE image's import and delay-import data directories, and the size of one descriptor in each.
_PE_IMPORTS, _PE_DELAY_IMPORTS = (1, 20), (13, 32)
# Descriptors read from one directory before the image counts as malformed; a real DLL imports far fewer modules.
_PE_DESCRIPTOR_LIMIT = 4096


def _pe_imports(read):
    """Names of the DLLs a PE image imports, directly or delay-loaded; () if they cannot be read.

    `read(rva, size)` returns `size` bytes of the image as the loader laid it out, from relative virtual address `rva`;
    it raises OSError for an address it cannot read."""
    def exact(rva, size):
        if rva < 0 or size < 0 or rva + size > image_size:
            raise ValueError("outside the image")
        data = read(rva, size)
        if len(data) != size:
            raise ValueError("short read")
        return data

    def name(rva):
        # A name with no NUL within 512 bytes or before the end of the image is malformed: the image is unverifiable.
        data = exact(rva, min(512, image_size - rva))
        end = data.find(b"\0")
        if end == -1:
            raise ValueError("unterminated name")
        return data[:end].decode("ascii")

    # Until the optional header gives SizeOfImage, reads are bounded by the headers being read.
    image_size = 0x40
    try:
        # The DOS header holds the offset of the PE signature, then the COFF header and the optional header.
        dos = exact(0, 0x40)
        if dos[:2] != b"MZ":
            return ()
        (pe,) = struct.unpack_from("<I", dos, 0x3C)
        image_size = pe + 24 + 64
        header = exact(pe, 24 + 64)
        if header[:4] != b"PE\0\0":
            return ()
        (magic,) = struct.unpack_from("<H", header, 24)
        (image_size,) = struct.unpack_from("<I", header, 24 + 56)
        # PE32+ (64-bit) and PE32 place the directory count and table at different offsets of the optional header.
        directories = {0x20B: 112, 0x10B: 96}.get(magic)
        if directories is None:
            return ()
        (count,) = struct.unpack_from("<I", exact(pe + 24 + directories - 4, 4))
        names = []
        for index, size in (_PE_IMPORTS, _PE_DELAY_IMPORTS):
            if index >= count:
                continue
            (start,) = struct.unpack_from("<I", exact(pe + 24 + directories + 8 * index, 4))
            if start == 0:
                continue
            for entry in range(_PE_DESCRIPTOR_LIMIT + 1):
                if entry == _PE_DESCRIPTOR_LIMIT:
                    return ()
                descriptor = exact(start + entry * size, size)
                if not any(descriptor):
                    break
                if index == _PE_IMPORTS[0]:
                    (rva,) = struct.unpack_from("<I", descriptor, 12)
                else:
                    attributes, rva = struct.unpack_from("<II", descriptor, 0)
                    # Only linkers older than Visual C++ 7 write a delay-import descriptor with virtual addresses.
                    if not attributes & 1:
                        return ()
                names.append(name(rva))
        return tuple(names)
    except (OSError, ValueError, struct.error, UnicodeDecodeError):
        return ()


def _kernel32():
    """kernel32 with the prototypes the Windows module checks call, in a library object of their own."""
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    for function, restype, argtypes in (
            ("GetCurrentProcess", wintypes.HANDLE, ()),
            ("K32EnumProcessModulesEx", wintypes.BOOL, (wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE),
                                                        wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                                        wintypes.DWORD)),
            ("GetModuleFileNameW", wintypes.DWORD, (wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD)),
            ("GetModuleHandleExW", wintypes.BOOL, (wintypes.DWORD, ctypes.c_void_p,
                                                   ctypes.POINTER(wintypes.HMODULE))),
            ("ReadProcessMemory", wintypes.BOOL, (wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                                  ctypes.POINTER(ctypes.c_size_t))),
            ("FreeLibrary", wintypes.BOOL, (wintypes.HMODULE,))):
        getattr(kernel32, function).restype = restype
        getattr(kernel32, function).argtypes = argtypes
    return kernel32


def _module_path(kernel32, module):
    """The file a loaded module was loaded from; OSError if it is not loaded."""
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel32.GetModuleFileNameW(module, buffer, len(buffer))
    if length == 0 or length >= len(buffer):
        raise OSError(ctypes.get_last_error(), "GetModuleFileNameW failed")
    return buffer.value


def _loaded_modules():
    """(file path, base address) of each module loaded in this Windows process; OSError if they cannot be listed."""
    from ctypes import wintypes
    kernel32 = _kernel32()
    process, needed, count = kernel32.GetCurrentProcess(), wintypes.DWORD(), 1024
    while True:
        modules = (wintypes.HMODULE * count)()
        # LIST_MODULES_ALL (3): 32-bit and 64-bit modules alike.
        if not kernel32.K32EnumProcessModulesEx(process, modules, ctypes.sizeof(modules), ctypes.byref(needed), 3):
            raise OSError(ctypes.get_last_error(), "K32EnumProcessModulesEx failed")
        if needed.value <= ctypes.sizeof(modules):
            break
        count = needed.value // ctypes.sizeof(wintypes.HMODULE) + 64
    return [(_module_path(kernel32, module), module)
            for module in modules[:needed.value // ctypes.sizeof(wintypes.HMODULE)]]


def _module_imports(path, base):
    """Names of the DLLs that the module `path` loaded at `base` imports, read from its image in this process's memory;
    () if it cannot be read.

    The module is pinned while it is read, so it cannot unload; one that has already unloaded, or that another module
    has replaced at `base`, reads as ()."""
    from ctypes import wintypes
    kernel32 = _kernel32()
    pinned = wintypes.HMODULE()
    # GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS (4): the module that holds the address `base`, with its count raised.
    if not kernel32.GetModuleHandleExW(4, base, ctypes.byref(pinned)):
        return ()
    try:
        if pinned.value != base or _module_path(kernel32, pinned) != path:
            return ()
        process = kernel32.GetCurrentProcess()

        def read(rva, size):
            # ReadProcessMemory reports a page that cannot be read, where a direct read would fault.
            buffer, done = ctypes.create_string_buffer(size), ctypes.c_size_t()
            if not kernel32.ReadProcessMemory(process, base + rva, buffer, size, ctypes.byref(done)):
                raise OSError(ctypes.get_last_error(), "ReadProcessMemory failed")
            return buffer.raw[:done.value]
        return _pe_imports(read)
    except OSError:
        return ()
    finally:
        kernel32.FreeLibrary(pinned)


def _loaded_usd_profile(modules=_loaded_modules, imports=_module_imports):
    """The Live Services USD profile among this Windows process's loaded modules, or None when there is none or it is
    ambiguous. `modules` lists (path, base) of each loaded module, and `imports(path, base)` names one's imports."""
    try:
        loaded = list(modules())
    except OSError:
        return None
    found = {profile: [(path, base) for path, base in loaded if PureWindowsPath(path).name.lower() == library]
             for profile, library in _HOST_USD_MODULES.items()}
    present = [profile for profile, copies in found.items() if copies]
    if len(present) == 1:
        return present[0]
    if len(present) == 2:
        # As on Linux, Kit 111's USD namespace shim, an ov_25.11usd_ms.dll that imports usd_sdf.dll, forwards to the
        # stock split USD this process already loaded. Every loaded copy must be that shim, read from its image in
        # memory: a real monolithic DLL next to stock USD, or one that cannot be read, is ambiguous.
        if all("usd_sdf.dll" in (name.lower() for name in imports(path, base)) for path, base in found["kit-25.11"]):
            return "stock-25.11"
    return None


def _check_usd_profile(package):
    """Reject a package for another USD ABI than the host's, before registration or import."""
    profile = json.loads((package / "config/live-sdk.json").read_text())
    if (not isinstance(profile, dict) or set(profile) != {"schema_version", "usd_profile"} or
            type(profile["schema_version"]) is not int or profile["schema_version"] != 1 or
            not isinstance(profile["usd_profile"], str) or profile["usd_profile"] not in _HOST_USD_MARKERS):
        raise RuntimeError("Kit requires an explicit kit-25.11 or stock-25.11 Live Services package")
    if _verifies_host_usd():
        host = _host_usd_profile()
        if host != profile["usd_profile"]:
            raise RuntimeError(f"Live Services package targets {profile['usd_profile']}, but this Kit process loaded "
                               f"{host or 'an unrecognized'} USD")
    elif profile["usd_profile"] != "kit-25.11":
        raise RuntimeError("Kit accepts a stock-25.11 Live Services package only on Linux and Windows, where it can "
                           "verify the host's USD; use the kit-25.11 package")


def _check_python_abi(package):
    """Reject a package built for another Python ABI before registration or import."""
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    # Every native extension the package holds, of any platform: one for another ABI is a mismatch.
    built = sorted(path.name for path in (package / "python/omni/usd/live").glob("_live.*")
                   if path.suffix in (".so", ".pyd"))
    if not suffix or built != ["_live" + suffix]:
        raise RuntimeError(f"Live Services package provides Python extension {built or 'none'}; "
                           f"this Python requires _live{suffix}")


def _check_plugins(package, registry):
    """Reject competing live handlers before registering this package's metadata."""
    package = Path(package).resolve()
    library = package / _LIBRARY
    resources = package / "resources"
    if not library.is_file() or not (resources / "plugInfo.json").is_file():
        raise RuntimeError("Live Services package is incomplete")
    for plugin in registry.GetAllPlugins():
        ours = plugin.name == "omni_usd_live" and Path(plugin.path).resolve() == library
        if plugin.name == "omni_usd_live" and not ours:
            raise RuntimeError("A different Live Services package is already registered")
        for metadata in plugin.metadata.get("Types", {}).values():
            if ("live" in metadata.get("extensions", ()) or
                    "live" in metadata.get("uriSchemes", ())):
                if not ours:
                    raise RuntimeError("A competing live file format or URI resolver is registered")
    return library


def register_plugin(package, registry):
    package = Path(package).resolve()
    library = _check_plugins(package, registry)
    registry.RegisterPlugins(str(package / "resources"))
    plugin = registry.GetPluginWithName("omni_usd_live")
    if plugin is None or Path(plugin.path).resolve() != library:
        raise RuntimeError("Live Services plugin registration did not select this package")
    return plugin


def load_api(package, registry):
    package = Path(package).resolve()
    _check_usd_profile(package)
    _check_python_abi(package)
    # Registration is metadata-only; reject competing handlers before importing
    # the candidate DSO. Importing a host USD parent may register more plugins,
    # so recheck after that import and before loading omni.usd.live.
    _check_plugins(package, registry)
    parent = importlib.import_module("omni.usd")
    _check_plugins(package, registry)
    # A namespace package has no __file__: this package's python/ directory
    # would own omni.usd, and Kit's regular omni.usd could never replace it.
    if getattr(parent, "__file__", None) is None:
        raise RuntimeError("omni.usd resolved to a namespace package; Kit's omni.usd must load before Live Services")
    expected = package / "python/omni/usd/live/__init__.py"
    existing = sys.modules.get("omni.usd.live")
    if existing is not None:
        if Path(existing.__file__).resolve() != expected:
            raise RuntimeError("A different Live Services Python package is already loaded")
        register_plugin(package, registry)
        return existing
    spec = importlib.util.find_spec("omni.usd.live")
    if spec is not None and (not spec.origin or Path(spec.origin).resolve() != expected):
        raise RuntimeError("A different Live Services Python package is on the host search path")
    register_plugin(package, registry)
    # Kit can already own a regular omni.usd package. Contribute only this
    # canonical child directory, retaining the parent's existing path object.
    contribution = str(expected.parent.parent)
    if contribution not in parent.__path__:
        parent.__path__.append(contribution)
    api = importlib.import_module("omni.usd.live")
    if Path(api.__file__).resolve() != expected:
        raise RuntimeError("Live Services import did not select this package")
    return api
