# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the package and USD registry before importing native Live Services."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import sysconfig


# The plugin library within the package: beside the samples on Windows, where the loader finds an imported DLL, and
# under lib/ elsewhere.
_LIBRARY = ("bin/omni_usd_live.dll" if sys.platform == "win32" else
            "lib/libomni_usd_live.dylib" if sys.platform == "darwin" else "lib/libomni_usd_live.so")


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
    profile = json.loads((package / "config/live-sdk.json").read_text())
    if (not isinstance(profile, dict) or type(profile.get("schema_version")) is not int or
            profile != {"schema_version": 1, "usd_profile": "kit-25.11"}):
        raise RuntimeError("Kit requires the explicit kit-25.11 Live Services package")
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
