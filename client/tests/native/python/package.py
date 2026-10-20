# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Relocate the actual Python/shared-plugin package and check one-runtime imports."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PROGRAM = r'''
import ctypes, gc, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
package=Path(os.environ['LIVE_PACKAGE'])
windows=os.name=='nt'
library=package/('bin/omni_usd_live.dll' if windows else
                 'lib/libomni_usd_live.dylib' if sys.platform=='darwin' else 'lib/libomni_usd_live.so')
# On Windows the standard search order (winmode=0) finds the SDK DLLs on PATH, which Python's default omits.
load=(lambda path: ctypes.CDLL(str(path),winmode=0)) if windows else (lambda path: ctypes.CDLL(str(path)))
if os.environ['LIVE_ORDER']=='usd-first':
    from pxr import Sdf
if os.environ['LIVE_ORDER']=='c-first':
    dll=load(library)
from omni.usd import live
def close_creation(creation):
    result=creation.close()
    assert result in (live.Result.OK, live.Result.BUSY),result
    if result==live.Result.BUSY:
        # Even an unstarted seed can have a worker still retiring admission.
        # Wait off-host on its explicit close notification, then release USD
        # ownership on the original host. These budgets are failure watchdogs.
        with ThreadPoolExecutor(max_workers=1) as worker:
            assert worker.submit(creation.wait_closed,3000).result(timeout=5)==live.Result.OK
        assert creation.close()==live.Result.OK
assert Path(live.__file__).resolve()==package/'python/omni/usd/live/__init__.py'
if os.environ['LIVE_ORDER']!='c-first':
    dll=load(library)
from pxr import Sdf, Plug
sdk_python=Path(os.environ['LIVE_USD_PYTHON'])
assert Path(Sdf.__file__).resolve().is_relative_to(sdk_python.resolve()), Sdf.__file__
Plug.Registry().RegisterPlugins(str(package/'resources'))
version=ctypes.c_uint32()
assert dll.omniUsdLiveGetAbiVersion(ctypes.byref(version))==0
assert version.value==live.get_abi_version()
runtime=live.initialize()
endpoint=runtime.register_endpoint('localhost:43123',access_token='synthetic')
token=dll.omniUsdLiveSetAccessToken
token.argtypes=[ctypes.c_uint64,ctypes.c_uint64,ctypes.c_char_p,ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64)]
token.restype=ctypes.c_uint32
generation=live.get_auth_state(endpoint).generation
replacement=ctypes.c_uint64()
assert token(endpoint.handle,generation,b'synthetic-new',0,ctypes.byref(replacement))==0
assert live.get_auth_state(endpoint).generation==replacement.value>generation
# Passing a real SDK SdfLayer through the pybind11/Boost.Python bridge must
# succeed. Freezing a creation seed is local: do not start a service request.
pending=[]
context=live.set_publication_context(pending.append)
seed=Sdf.Layer.CreateAnonymous('relocated-seed.usda')
seed.startTimeCode=17.0
uri=live.make_layer_uri(live.make_session_uri('localhost:43123','relocated'), 'seed.live')
creation=context.prepare_creation(seed,uri)
close_creation(creation)
assert context.close(3000)==live.Result.OK
assert not pending
if windows:
    # The process holds one omni_usd_live.dll, the package's.
    kernel32=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel32.GetModuleHandleW.restype=ctypes.c_void_p
    kernel32.GetModuleFileNameW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_uint32]
    module=kernel32.GetModuleHandleW('omni_usd_live.dll')
    name=ctypes.create_unicode_buffer(32768)
    assert module and kernel32.GetModuleFileNameW(module,name,len(name))
    assert os.path.samefile(name.value,library),name.value
elif sys.platform=='darwin':
    dyld=ctypes.CDLL(None)
    dyld._dyld_image_count.restype=ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes=[ctypes.c_uint32]
    dyld._dyld_get_image_name.restype=ctypes.c_char_p
    images={Path(os.fsdecode(dyld._dyld_get_image_name(index))).resolve()
            for index in range(dyld._dyld_image_count())}
    matches={path for path in images if path.name=='libomni_usd_live.dylib'}
    assert matches=={library.resolve()},matches
else:
    maps={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libomni_usd_live.so' in line}
    assert maps=={str(library)},maps
assert runtime.close(3000)==live.Result.OK
# Exercise the Kit lifecycle owner with the real SDK/API. This is not a Kit
# application test: a prepared, unstarted creation needs no host scheduling.
from omni.usd.live._kit_host import KitHost
from omni.usd.live import _kit_root
assert Path(_kit_root.__file__).resolve()==package/'python/omni/usd/live/_kit_root.py'
import importlib.util
spec=importlib.util.spec_from_file_location('live_bootstrap',package/'python/omni/kit/live_services/_bootstrap.py')
bootstrap=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
assert bootstrap.register_plugin(package, Plug.Registry()).name=='omni_usd_live'
def unexpected_publication(coroutine):
    coroutine.close()
    raise AssertionError('unstarted creation scheduled host publication')
host=KitHost(unexpected_publication)
host.start()
registration=host.register_endpoint('localhost:43123', access_token='synthetic-kit',
    on_auth_required=lambda event: None, on_status=lambda event: None)
creation=host.publication_context.prepare_creation(seed,uri)
assert host.close()==live.Result.BUSY
assert registration.auth.delivery_status==live.Result.OK
assert registration.status.delivery_status==live.Result.OK
auth=live.get_auth_state(registration.endpoint)
assert live.set_access_token(registration.endpoint,auth.generation,'synthetic-refresh')[0]==live.Result.OK
close_creation(creation)
assert host.close(3000)==live.Result.OK
gc.collect()
print('relocated Python/C ownership, SDK Sdf conversion and Kit owner cleanup passed',os.environ['LIVE_ORDER'])
'''


def main():
    package = Path(sys.argv[1]).resolve()
    sdk_python = Path(sys.argv[2]).resolve() / 'lib/python'
    assert (sdk_python / 'pxr/Sdf/__init__.py').is_file()
    windows = os.name == "nt"
    extension = list((package / "python/omni/usd/live").glob("_live*.pyd" if windows else "_live*.so"))
    assert len(extension) == 1
    assert not any(p.suffix in {".pyc", ".pyo"} for p in (package / "python").rglob("*"))
    if windows:
        dynamic = subprocess.check_output(["dumpbin", "/nologo", "/DEPENDENTS", str(extension[0])], text=True)
        assert "omni_usd_live.dll" in dynamic.lower() and "omniclient" not in dynamic.lower()
    elif sys.platform == "darwin":
        dynamic = subprocess.check_output(["otool", "-L", str(extension[0])], text=True)
        assert "@rpath/libomni_usd_live.dylib" in dynamic and "omniclient" not in dynamic.lower()
    else:
        dynamic = subprocess.check_output(["readelf", "-d", str(extension[0])], text=True)
        assert "libomni_usd_live.so" in dynamic and "omniclient" not in dynamic.lower()
    with tempfile.TemporaryDirectory(prefix="live-python-relocated-") as directory:
        copied = Path(directory).resolve() / "plugin"
        shutil.copytree(package, copied)
        for order in ("c-first", "python-first", "usd-first"):
            environment = dict(os.environ, LIVE_PACKAGE=str(copied), LIVE_ORDER=order,
                               LIVE_USD_PYTHON=str(sdk_python),
                               PYTHONPATH=os.pathsep.join((str(copied / "python"), str(sdk_python))), PYTHONNOUSERSITE="1",
                               PYTHONDONTWRITEBYTECODE="1", PXR_PLUGINPATH_NAME="")
            subprocess.run([sys.executable, "-c", PROGRAM], env=environment,
                           check=True, timeout=15, cwd=directory)


if __name__ == "__main__":
    main()
