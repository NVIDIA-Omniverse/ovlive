# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual Kit/UsdContext participant, controlled by the reference-service fixture."""
import argparse
import asyncio
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import sysconfig
import threading
import time
import traceback


def fields(layer, prim='/World'):
    return {key: layer.GetAttributeAtPath(prim + '.' + key).default for key in ('a', 'b', 'c')}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


async def main(args):
    import carb.settings
    import omni.kit.app
    import omni.usd
    from omni.kit.live_services import get_host, close_host
    from omni.usd import live
    from pxr import Plug, Sdf, Tf, Usd, UsdUtils
    app = omni.kit.app.get_app()
    report = {'passed': False, 'pid': os.getpid(), 'stage_root': args.stage_root}
    writer = None
    notice = None
    diagnostic_sink = None
    diagnostic_release = threading.Event()
    diagnostic_exited = threading.Event()

    def phase(name):
        report['phase'] = name
        report['phase_time_ns'] = time.monotonic_ns()
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('ovlive Kit phase: ' + name, flush=True)

    try:
        phase('preflight')
        if not __debug__:
            raise RuntimeError('Kit qualification requires enabled Python assertions')
        if not sys.dont_write_bytecode:
            raise RuntimeError('Kit qualification requires disabled bytecode writes')
        package = args.package.resolve()
        settings = carb.settings.get_settings()
        required_settings = {'/exts/omni.usd/locking/earlyRelease': True,
                             '/exts/omni.usd/locking/onClose': True,
                             '/app/useFabricSceneDelegate': False}
        observed_settings = {key: settings.get(key) for key in required_settings}
        assert observed_settings == required_settings, observed_settings
        manager = app.get_extension_manager()
        extension_configs = {}
        selected_extensions = [('omni_usd_live', package), ('omni.client', args.client_extension.resolve()),
                               ('omni.usd_resolver', args.resolver_extension.resolve())]
        if args.core_extension:
            selected_extensions.append(('omni.usd.core', args.core_extension.resolve()))
        for name, expected in selected_extensions:
            ext_id = manager.get_enabled_extension_id(name)
            assert ext_id and Path(manager.get_extension_path(ext_id)).resolve() == expected, name
            extension_configs[name] = digest(expected / 'config/extension.toml')
        assert manager.get_enabled_extension_id('omni.usd')
        assert Path(live.__file__).resolve() == package / 'python/omni/usd/live/__init__.py'
        plugin = Plug.Registry().GetPluginWithName('omni_usd_live')
        library = package / 'lib/libomni_usd_live.so'
        assert plugin is not None and Path(plugin.path).resolve() == library
        mapped = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                  if '/' in line.split()[-1]}
        assert {p for p in mapped if p.endswith('/libomni_usd_live.so')} == {str(library)}
        if args.core_extension:
            tbb = (args.core_extension / 'bin/libomni.tbb.globalcontrol.plugin.so').resolve(strict=True)
            assert {p for p in mapped if p.endswith('/libomni.tbb.globalcontrol.plugin.so')} == {str(tbb)}
        for required in (args.client_extension.resolve() / 'bin/libomni.client.lifecycle.plugin.so',
                         args.resolver_extension.resolve() / 'lib/libcarb.usdresolver.plugin.so',
                         args.resolver_extension.resolve() / 'lib/libomni_usd_resolver.so',
                         args.resolver_extension.resolve() / 'bin/libomni.ext-usd_resolver.plugin.so'):
            assert str(required) in mapped, 'Required lifecycle/resolver library not loaded: ' + str(required)
        assert not any('omni.client.liveprocess' in p or 'spectree' in Path(p).name for p in mapped)
        # Kit keeps an inert omni.spectree Python namespace for import compatibility. What must not exist is
        # another native implementation or another plugin that claims the live file format.
        claimants = sorted(plugin.name for plugin in Plug.Registry().GetAllPlugins()
                           if any('live' in (info.get('extensions') or [])
                                  for info in (plugin.metadata.get('Types') or {}).values()))
        assert claimants == ['omni_usd_live'], 'Another plugin claims the live file format: ' + repr(claimants)
        live_format = Sdf.FileFormat.FindByExtension('live')
        assert live_format is not None and live_format.formatId == 'live'
        report.update(python_version=sys.version, python_optimize=sys.flags.optimize,
                      dont_write_bytecode=sys.dont_write_bytecode, settings=observed_settings,
                      driver_sha256=digest(__file__), executable_sha256=digest('/proc/self/exe'),
                      extension_configs=extension_configs,
                      native_libraries={p: digest(p) for p in sorted(mapped) if
                          any(name in Path(p).name for name in
                              ('usd', 'OpenGL', 'python3.12', 'omni.client.lifecycle', 'omni.tbb.globalcontrol'))})
        adapter = Path(live._live.__file__).resolve()
        assert adapter.parent == package / 'python/omni/usd/live', adapter
        report.update(python_abi=dict(soabi=sysconfig.get_config_var('SOABI'),
                                      ext_suffix=sysconfig.get_config_var('EXT_SUFFIX'),
                                      cache_tag=sys.implementation.cache_tag),
                      live_module=dict(path=str(adapter), sha256=digest(adapter), abi_version=live.get_abi_version()))
        host = get_host()
        assert host is not None
        creator = threading.get_ident()
        loop = asyncio.get_running_loop()
        status_changed, layer_changed = asyncio.Event(), asyncio.Event()
        status_events = []
        exported_file = None
        notice_threads = []
        uri = live.make_layer_uri(live.make_session_uri(args.authority, args.session), 'root.live')
        healthy_uri = live.make_layer_uri(live.make_session_uri(args.authority, args.session), 'healthy.live')
        healthy = None
        diagnostic_entered = asyncio.Event()
        diagnostic_calls = 0

        def status(event):
            if event.uri in (uri, healthy_uri):
                # Copy only bounded test evidence; never touch USD on this consumer.
                def delivered():
                    if len(status_events) >= 1024:
                        raise RuntimeError('status evidence overflow')
                    status_events.append(asdict(event.status))
                    status_changed.set()
                loop.call_soon_threadsafe(delivered)

        registration = host.register_endpoint(args.authority, access_token='synthetic-writer',
            on_auth_required=lambda event: None, on_status=status)
        phase('opening-live-layer')
        layer = Sdf.Layer.FindOrOpen(uri)
        assert layer and layer.permissionToEdit
        phase('waiting-for-layer-ready')
        while True:
            status_changed.clear()
            current = live.get_layer_status(uri)
            if current.state == live.State.READY:
                break
            assert current.state != live.State.FAILED, current
            await status_changed.wait()
        def changed(event, sender):
            # Global notices include every layer, such as the reopened export.
            # Only the attached live layers belong to this host's
            # publication/thread contract.
            if layer not in event.GetLayers() and (healthy is None or healthy not in event.GetLayers()):
                return
            if len(notice_threads) >= 1024:
                raise RuntimeError('notice evidence overflow')
            notice_threads.append(threading.get_ident())
            layer_changed.set()

        notice = Tf.Notice.RegisterGlobally(Sdf.Notice.LayersDidChange, changed)
        phase('composing-stage')
        base = session = None
        if args.stage_root == 'live':
            stage = Usd.Stage.Open(layer)
        else:
            base = Sdf.Layer.CreateAnonymous('kit-reference-base.usda')
            session = Sdf.Layer.CreateAnonymous('kit-reference-session.usda')
            session.subLayerPaths = [uri]
            stage = Usd.Stage.Open(base, session)
        stage_cache = UsdUtils.StageCache.Get()
        assert not stage_cache.Contains(stage)
        context = omni.usd.get_context()
        phase('attaching-stage')
        ok, error = await context.attach_stage_async(stage)
        assert ok, error
        assert context.get_stage() == stage and layer in stage.GetLayerStack()
        # attach_stage_async inserts this stage into the caller-owned cache.
        # The context deliberately leaves pre-existing entries to their owner.
        assert stage_cache.Contains(stage)
        if args.stage_root == 'live':
            assert stage.GetRootLayer() == layer and layer.permissionToEdit
            report['initialized_status'] = asdict(live.get_layer_status(uri))
            report['initialized_root_fields'] = sorted(layer.pseudoRoot.ListInfoKeys())
        phase('opening-control-channel')
        reader, writer = await asyncio.open_connection('127.0.0.1', args.control_port, limit=65536)

        async def send(value):
            encoded = json.dumps(value).encode() + b'\n'
            assert len(encoded) < 65536
            writer.write(encoded)
            await writer.drain()

        def attachment(target):
            assert target in ('root', 'healthy')
            if target == 'root':
                return layer, uri, '/World'
            assert healthy is not None
            return healthy, healthy_uri, '/Healthy'

        def state(target='root'):
            assert threading.get_ident() == creator
            selected, selected_uri, prim = attachment(target)
            assert selected.permissionToEdit
            current = live.get_layer_status(selected_uri)
            return dict(fields=fields(selected, prim),
                        stage={key: stage.GetAttributeAtPath(prim + '.' + key).Get() for key in ('a', 'b', 'c')},
                        pending=current.pending_write_count, subject=current.subject_id,
                        status=asdict(current),
                        delivery=int(host.publication_context.delivery_status))

        def edit(target, values):
            selected, _, prim = attachment(target)
            with Sdf.ChangeBlock():
                for key, value in values.items():
                    assert key in ('a', 'b', 'c')
                    selected.GetAttributeAtPath(prim + '.' + key).default = value

        async def save(selected):
            # An off-host save returns BUSY while host edits await admission;
            # the Kit loop runs the queued flush job before the next attempt.
            # The deadline is a failure-only watchdog.
            deadline = time.monotonic() + 10
            while (result := await asyncio.to_thread(live.save_layer, selected, 10000)) == live.Result.BUSY:
                assert time.monotonic() < deadline, 'save stayed BUSY'
                await app.next_update_async()
            return result

        def root_metadata():
            assert threading.get_ident() == creator and args.stage_root == 'live'
            result = {}
            for key in layer.pseudoRoot.ListInfoKeys():
                if key == 'primChildren':
                    continue
                value = layer.pseudoRoot.GetInfo(key)
                if key in ('startTimeCode', 'endTimeCode', 'framesPerSecond', 'timeCodesPerSecond', 'metersPerUnit'):
                    assert type(value) is float, (key, value)
                    result[key] = [9, value]
                else:
                    assert key in ('upAxis', 'defaultPrim') and type(value) is str, (key, value)
                    result[key] = [11, value]
            return result

        await send(dict(type='ready', nonce=args.nonce, **state()))
        phase('processing-commands')
        last_id = 0
        while raw := await reader.readline():
            command = json.loads(raw)
            identity, kind = command['id'], command['type']
            assert type(identity) is int and identity > last_id
            last_id = identity
            report['current_command'] = dict(id=identity, type=kind)
            phase('command-' + kind)
            target = command.get('target', 'root')
            assert target == 'root' or kind in ('edit', 'save', 'wait', 'state')
            if kind == 'root-host-edit':
                assert args.stage_root == 'live' and layer.permissionToEdit
                # Ordinary host authoring on the editable attached live root.
                # The test supplies no permission override or application guard.
                # Keep a valid range: Kit's timeline intentionally adjusts an
                # end time at/before its start when processing USD notices.
                with Sdf.ChangeBlock():
                    layer.endTimeCode = 100.0
                    layer.startTimeCode = 37.0
                assert layer.startTimeCode == 37.0 and stage.GetStartTimeCode() == 37.0
                assert layer.endTimeCode == 100.0 and stage.GetEndTimeCode() == 100.0
                # The save is the progress boundary: OK means the service
                # confirmed every edit authored above.
                assert await save(layer) == live.Result.OK
                assert registration.status.delivery_status == live.Result.OK
                metadata = root_metadata()
                assert metadata['startTimeCode'] == [9, 37.0] and metadata['endTimeCode'] == [9, 100.0]
                reply = dict(start_time_code=layer.startTimeCode, end_time_code=layer.endTimeCode,
                             root_fields=metadata, **state())
            elif kind == 'open-healthy':
                assert healthy is None
                healthy = Sdf.Layer.FindOrOpen(healthy_uri)
                assert healthy and healthy.permissionToEdit
                while True:
                    status_changed.clear()
                    current = live.get_layer_status(healthy_uri)
                    if current.state == live.State.READY:
                        break
                    assert current.state != live.State.FAILED, current
                    await status_changed.wait()
                session.subLayerPaths.append(healthy_uri)
                assert healthy in stage.GetLayerStack()
                reply = state('healthy')
            elif kind == 'hold-diagnostics':
                assert diagnostic_sink is None
                def held(record):
                    nonlocal diagnostic_calls
                    diagnostic_calls += 1
                    if diagnostic_calls == 1:
                        loop.call_soon_threadsafe(diagnostic_entered.set)
                        diagnostic_release.wait()
                        diagnostic_exited.set()
                diagnostic_sink = live.register_diagnostic_callback(held, endpoint=registration.endpoint,
                    min_severity=live.Severity.DEBUG, max_records=1, max_bytes=4096)
                # A finite read guarantees a diagnostic event without introducing
                # another mutation or relying on background traffic timing.
                result, _ = await live.list_sessions_async(registration.endpoint, 'diagnostic-trigger.usda')
                assert result == live.Result.OK
                await diagnostic_entered.wait()
                assert not diagnostic_exited.is_set()
                reply = dict(held=True)
            elif kind == 'diagnostics-state':
                assert diagnostic_sink is not None and diagnostic_entered.is_set()
                assert not diagnostic_exited.is_set()
                reply = dict(held=True, calls=diagnostic_calls, dropped=diagnostic_sink.dropped_records)
            elif kind == 'release-diagnostics':
                assert diagnostic_sink is not None and diagnostic_entered.is_set()
                assert not diagnostic_exited.is_set()
                diagnostic_release.set()
                assert await asyncio.to_thread(diagnostic_sink.close, 5000) == live.Result.OK
                assert diagnostic_exited.is_set()
                report['diagnostic_isolation'] = dict(entered=True, retired=True, calls=diagnostic_calls)
                diagnostic_sink = None
                reply = dict(retired=True)
            elif kind == 'edit':
                edit(target, command['fields'])
                reply = dict(authored=True)
            elif kind == 'save':
                assert await save(attachment(target)[0]) == live.Result.OK
                reply = state(target)
            elif kind == 'wait':
                while True:
                    layer_changed.clear()
                    observed = state(target)
                    if observed['fields'] == command['fields']:
                        for key, value in command['fields'].items():
                            assert observed['stage'][key] == value, observed
                        break
                    assert host.publication_context.delivery_status == live.Result.OK
                    await layer_changed.wait()
                reply = observed
            elif kind == 'state':
                reply = state(target)
            elif kind == 'failed':
                while True:
                    status_changed.clear()
                    current = live.get_layer_status(uri)
                    matching = [event for event in status_events if
                                event['subject_id'] == current.subject_id and
                                event['state'] == live.State.FAILED and
                                event['result'] == live.Result.OUTCOME_UNKNOWN and
                                event['incident_id'] == current.incident_id]
                    if current.state == live.State.FAILED and matching:
                        assert current.result == live.Result.OUTCOME_UNKNOWN
                        assert current.has_unconfirmed_edits and current.incident_id
                        assert registration.status.delivery_status == live.Result.OK
                        break
                    await status_changed.wait()
                # The typed incident must refuse a save. No diagnostic
                # subscription or log text participates in this decision.
                assert await save(layer) == live.Result.OUTCOME_UNKNOWN
                reply = dict(observed_status=matching[-1], **state())
            elif kind == 'export':
                before = live.get_layer_status(uri)
                assert before.state == live.State.FAILED
                path = args.output.parent / 'retained.usda'
                assert not path.exists()
                # SdfLayer::Export of the failed layer on Kit's main thread (design section 13).
                assert layer.Export(str(path))
                assert live.get_layer_status(uri) == before
                exported_file = (path, digest(path), before.subject_id, before.incident_id)
                reply = dict(sha256=exported_file[1],
                             exported=fields(Sdf.Layer.OpenAsAnonymous(str(path))), **state())
            elif kind == 'close':
                assert diagnostic_sink is None, 'retire the held diagnostic callback explicitly'
                if healthy is not None:
                    assert await save(healthy) == live.Result.OK
                    report['healthy_final'] = state('healthy')
                phase('closing-stage')
                if command.get('exported') is True:
                    current = live.get_layer_status(uri)
                    assert exported_file is not None and exported_file[2:] == (current.subject_id, current.incident_id)
                    assert current.state == live.State.FAILED and digest(exported_file[0]) == exported_file[1]
                    report['export_sha256'] = exported_file[1]
                else:
                    assert await save(layer) == live.Result.OK
                ok, error = await context.close_stage_async()
                assert ok, error
                phase('releasing-stage-cache-entry')
                assert stage_cache.Erase(stage)
                notice.Revoke()
                notice = None
                phase('releasing-usd-owners')
                stage = layer = healthy = session = base = None
                phase('collecting-python-owners')
                gc.collect()
                phase('closing-host')
                # Native capture retirement may be queued on this exact Kit loop.
                # Its update callback provides progress; only typed OK proves close.
                while True:
                    closed = close_host(5000)
                    if closed == live.Result.OK:
                        break
                    assert closed in (live.Result.BUSY, live.Result.TIMEOUT), closed
                    await app.next_update_async()
                assert get_host() is None
                assert notice_threads and set(notice_threads) == {creator}, notice_threads
                report.update(passed=True, notices=len(notice_threads), host_thread=creator,
                              status_events=status_events, commands=last_id)
                await send(dict(id=identity, type='closed'))
                break
            else:
                raise AssertionError(kind)
            await send(dict(id=identity, type=kind, **reply))
            phase('command-completed')
        assert report['passed'], 'control connection ended before explicit retirement'
    except BaseException:
        report['passed'] = False
        report['error'] = traceback.format_exc()
    finally:
        diagnostic_release.set()
        if diagnostic_sink is not None:
            retired = await asyncio.to_thread(diagnostic_sink.close, 5000)
            if retired != live.Result.OK:
                report['passed'] = False
                report['diagnostic_close_result'] = int(retired)
        if notice is not None:
            notice.Revoke()
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                report['passed'] = False
                report['close_error'] = traceback.format_exc()
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        app.post_quit(0 if report['passed'] else 1)


def entry():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'client-extension', 'resolver-extension', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('authority', 'session', 'nonce'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--control-port', required=True, type=int)
    parser.add_argument('--core-extension', type=Path)
    parser.add_argument('--stage-root', choices=('composed', 'live'), default='composed')
    args = parser.parse_args()
    from omni.kit.async_engine import run_coroutine
    run_coroutine(main(args))


if __name__ == '__main__':
    entry()
