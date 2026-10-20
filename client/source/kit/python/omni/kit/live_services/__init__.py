# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-reloadable Kit bootstrap; no native Live Services import at module load."""
import asyncio
from pathlib import Path
import sys

import carb
import omni.ext

from ._bootstrap import load_api

# Set on sys by the first start in a process. Kit removes this extension's modules from sys.modules when it shuts the
# extension down, and on a later start reloads omni.usd.live over the host that on_startup has started; the marker
# outlives those modules, so a second start fails before any import.
_STARTED = "_omni_kit_live_services_started"


def get_host():
    """Return the retained owner, including incomplete startup/shutdown owners."""
    api = sys.modules.get("omni.usd.live")
    return getattr(api, "_kit_owner", None)


def close_host(timeout_ms=0):
    """Call on the Kit main thread after preserving and detaching live layers."""
    api = sys.modules.get("omni.usd.live")
    host = get_host()
    if host is None:
        return 0  # OmniUsdLiveResult OK, without importing a native library.
    result = host.close(timeout_ms)
    if result == api.Result.OK:
        api._kit_owner = None
    return result


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        if getattr(sys, _STARTED, False):
            raise RuntimeError("Live Services starts once per Kit process; restart Kit to enable the extension again")
        setattr(sys, _STARTED, True)
        from pxr import Plug

        api = load_api(Path(__file__).resolve().parents[4], Plug.Registry())
        if get_host() is not None:
            raise api.LiveError(api.Result.BUSY)
        from omni.usd.live._kit_host import KitHost
        # Kit's loop runs on this main thread; the publication consumer thread
        # submits jobs. run_coroutine_threadsafe queues through
        # call_soon_threadsafe and returns a concurrent Future.
        loop = asyncio.get_event_loop()
        self._owner = KitHost(lambda coroutine: asyncio.run_coroutine_threadsafe(coroutine, loop))
        # Publish before allocating native owners: interrupted startup retains
        # a discoverable object for explicit cleanup by this instance or host.
        api._kit_owner = self._owner
        self._owner.start()

    def on_shutdown(self):
        owner = getattr(self, "_owner", None)
        if owner is None or owner is not get_host():
            return  # Failed startup must never close another instance's owner.
        result = close_host(0)
        if result != 0:
            carb.log_error("Live Services shutdown incomplete: " + result.name +
                           ". Preserve pending work and finish close_host before disabling the extension.")
