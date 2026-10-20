# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual KitHost/root workflow in an asyncio host, without a Kit application."""
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys

from pxr import Plug, Sdf
from omni.usd import live
from omni.usd.live._kit_host import KitHost


def emit(value):
    print(json.dumps(value), flush=True)


async def main():
    authority, package = sys.argv[1:3]
    Plug.Registry().RegisterPlugins(str(Path(package) / 'resources/plugInfo.json'))
    loop = asyncio.get_running_loop()
    host = KitHost(lambda coroutine: asyncio.run_coroutine_threadsafe(coroutine, loop))
    host.start(flags=1)
    registration = host.register_endpoint(authority, access_token='synthetic-writer',
        on_auth_required=lambda event: None, on_status=lambda event: None)
    catalog = host.session_catalog(registration, 'kit-root-stable-key', current_user='writer')
    root = operation = None
    emit(dict(type='ready'))
    while line := await asyncio.to_thread(sys.stdin.readline):
        command = line.rstrip('\n').partition(' ')[0]
        if command == 'prepare':
            operation = catalog.create('kit root creation')
            result, created = await operation.wait()
            assert result == live.Result.OK
            assert await asyncio.to_thread(operation.request.wait, 5000) == live.Result.OK
            seed = Sdf.Layer.CreateAnonymous('kit-root-seed.usda')
            seed.startTimeCode = 17.0
            root = host.prepare_session_root(catalog, operation, seed)
            seed.startTimeCode = 99.0
            assert host.prepare_session_root(catalog, operation, seed) is root
            emit(dict(type='prepared', session=created.session_id, uri=root.root_uri,
                      outcome=asdict(root.creation.outcome)))
        elif command == 'start':
            emit(dict(type='started', result=int(root.start())))
        elif command == 'wait-cancel':
            entered = asyncio.Event()
            async def waiter():
                entered.set()
                return await root.wait(5000)
            task = asyncio.create_task(waiter())
            await entered.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError('UI waiter was not cancelled')
            assert root._wait_task is not None and not root._wait_task.cancelled()
            emit(dict(type='cancelled', close_result=int(root.close()), receipts=len(catalog.operations)))
        elif command == 'wait':
            result = await root.wait(5000)
            emit(dict(type='waited', result=int(result), outcome=asdict(root.creation.outcome)))
        elif command == 'activate':
            result = root.activate(5000)
            fields = None
            if result == live.Result.OK:
                fields = dict(startTimeCode=root.layer.startTimeCode)
            emit(dict(type='activated', result=int(result), fields=fields))
        elif command == 'close-busy':
            emit(dict(type='closing', result=int(host.close()), receipts=len(catalog.operations),
                      roots=len(host.session_roots), root_closed=root.closed))
        elif command == 'close':
            result = root.close()
            if result == live.Result.BUSY:
                assert await asyncio.to_thread(root.creation.wait_closed, 5000) == live.Result.OK
                result = root.close()
            assert result == live.Result.OK, result
            # A retained closed workflow is never permission to replace root.live.
            if not host._closing:
                assert host.prepare_session_root(catalog, operation, None) is root
            assert root.start() == live.Result.NOT_INITIALIZED
            assert catalog.acknowledge(operation) == live.Result.OK
            assert host.close(5000) == live.Result.OK
            emit(dict(type='closed'))
            return
        else:
            raise AssertionError('unknown driver command')
    raise AssertionError('driver input closed before explicit cleanup')


if __name__ == '__main__':
    asyncio.run(main())
