# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a real headless Kit bootstrap/dispatch/retirement check, without a service.

The selected Kit client extension must omit the live frame pump.
This checks the package lifecycle; it does not qualify stage/Fabric synchronization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path):
    with Path(path).open('rb') as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
        return checksum.hexdigest()


def launch(args):
    from dataclasses import asdict
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.live.test_support import run_process
    out = args.artifact_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    package = args.package.resolve()
    kit = args.kit_executable.resolve()
    client = args.client_extension.resolve()
    for path in (kit, package / 'config/extension.toml', client / 'config/extension.toml'):
        if not path.is_file():
            raise ValueError(f'Missing required input: {path}')
    app = out / 'live_bootstrap.kit'
    app.write_text('''[package]
version = "1.0.0"
[settings]
app.extensions.registryEnabled = false
app.extensions.fsWatcherEnabled = false
app.fastShutdown = false
app.uploadDumpsOnStartup = false
crashreporter.enabled = false
telemetry.mode = "test"
''')
    probe_args = [str(Path(__file__).resolve()), 'probe', '--package', str(package),
                  '--client-extension', str(client), '--artifact-dir', str(out)]
    # Kit parses --exec as a command string; quote each path for its parser.
    import shlex
    command = [str(kit), str(app), '--no-window', '--portable', '--portable-root', str(out / 'portable'),
               '--ext-path', str(client), '--ext-path', str(package), '--enable', 'omni.kit.loop',
               '--enable', 'omni_usd_live',
               '--exec', shlex.join(probe_args), '--/log/file=', '--/log/outputStreamLevel=Info',
               '--/plugins/carb.scripting-python.plugin/pythonFlags/Py_IsolatedFlag=0',
                   '--/plugins/carb.scripting-python.plugin/pythonFlags/Py_IgnoreEnvironmentFlag=0']
    (out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    env = {key: value for key, value in os.environ.items()
           if key in {'PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT', 'TEMP', 'TMP'}}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    completed = asdict(run_process(command, cwd=out, output=out / 'stdout.log',
                                  environment=env, timeout_ms=int(args.watchdog_seconds * 1000),
                                  max_output_bytes=4 * 1024 * 1024))
    (out / 'process.json').write_text(json.dumps(completed, indent=2) + '\n')
    (out / 'result.json').write_text(json.dumps({'passed': False, 'process': completed}, indent=2) + '\n')
    if (completed['outcome'] != 'passed' or completed['returncode'] != 0
            or completed['cleanup_complete'] is not True or completed['output_truncated'] is not False):
        raise RuntimeError(completed)
    if '[Error]' in (out / 'stdout.log').read_text(errors='replace'):
        raise RuntimeError('Kit reported an error; inspect stdout.log')
    report = json.loads((out / 'probe.json').read_text())
    if report.get('passed') is not True:
        raise RuntimeError(report)
    report['process'] = completed
    report['kit_executable_sha256'] = digest(kit)
    report['driver_sha256'] = digest(__file__)
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Real Kit package startup, dispatch and explicit retirement passed:', out)


def probe(args):
    import asyncio
    import threading
    import traceback
    import omni.kit.app
    from omni.kit.async_engine import run_coroutine

    app = omni.kit.app.get_app()

    async def check():
        report = {'passed': False}
        try:
            if not __debug__:
                raise RuntimeError('Kit qualification requires enabled Python assertions')
            if not sys.dont_write_bytecode:
                raise RuntimeError('Kit qualification requires disabled bytecode writes')
            from pxr import Plug, Sdf
            from omni.usd import live
            from omni.kit.live_services import get_host, close_host
            package = args.package.resolve()
            manager = app.get_extension_manager()
            for name, expected in (('omni_usd_live', package), ('omni.client', args.client_extension.resolve())):
                ext_id = manager.get_enabled_extension_id(name)
                assert ext_id, name
                assert Path(manager.get_extension_path(ext_id)).resolve() == expected, (name, ext_id)
            assert Path(live.__file__).resolve() == package / 'python/omni/usd/live/__init__.py'
            plugin = Plug.Registry().GetPluginWithName('omni_usd_live')
            library = package / 'lib/libomni_usd_live.so'
            assert plugin is not None and Path(plugin.path).resolve() == library
            mapped = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                      if '/' in line.split()[-1]}
            assert not any('omni.client.liveprocess' in path for path in mapped), 'Live frame pump loaded'
            lifecycle = args.client_extension.resolve() / 'bin/libomni.client.lifecycle.plugin.so'
            assert str(lifecycle) in mapped, 'Storage lifecycle replacement not loaded'
            import omni.client
            storage = args.artifact_dir / 'storage-roundtrip.bin'
            content = b'live-services-kit-storage-roundtrip'
            assert omni.client.write_file(str(storage), content) == omni.client.Result.OK
            result, _, loaded = omni.client.read_file(str(storage))
            assert result == omni.client.Result.OK and bytes(loaded) == content
            assert {p for p in mapped if p.endswith('/libomni_usd_live.so')} == {str(library)}
            host = get_host()
            assert host is not None and host.publication_context.delivery_status == live.Result.OK
            creator = threading.get_ident()
            applied = asyncio.Event()
            dispatch_threads = []

            def job():
                dispatch_threads.append(threading.get_ident())
                applied.set()

            def submit():
                assert threading.get_ident() != creator
                host._dispatch(job)

            await asyncio.to_thread(submit)
            await applied.wait()
            assert dispatch_threads == [creator]
            registration = host.register_endpoint('localhost:43123', access_token='synthetic-bootstrap',
                on_auth_required=lambda event: None, on_status=lambda event: None)
            seed = Sdf.Layer.CreateAnonymous('kit-bootstrap-seed.usda')
            seed.startTimeCode = 17.0
            uri = live.make_layer_uri(live.make_session_uri('localhost:43123', 'bootstrap'), 'root.live')
            creation = host.publication_context.prepare_creation(seed, uri)
            # No CreateLayer request is started. The real SDK seed must still
            # retain host ownership until its explicit retirement completes.
            assert host.close() == live.Result.BUSY
            assert registration.auth.delivery_status == live.Result.OK
            result = creation.close()
            if result == live.Result.BUSY:
                assert await asyncio.to_thread(creation.wait_closed, 5000) == live.Result.OK
                result = creation.close()
            assert result == live.Result.OK
            assert close_host(5000) == live.Result.OK
            assert get_host() is None
            report.update(passed=True, python_version=sys.version, python_optimize=sys.flags.optimize,
                          host_thread=creator, dispatched_thread=dispatch_threads[0],
                          package=str(package), client_extension=str(args.client_extension.resolve()),
                          plugin_sha256=digest(library), lifecycle_sha256=digest(lifecycle), abi_version=live.get_abi_version(),
                          usd_libraries={p: digest(p) for p in sorted(mapped) if p.endswith('usd_ms.so')})
        except BaseException:
            report['error'] = traceback.format_exc()
        finally:
            (args.artifact_dir / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')
            app.post_quit(0 if report['passed'] else 1)

    run_coroutine(check())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'probe'))
    parser.add_argument('--kit-executable', type=Path)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--client-extension', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--watchdog-seconds', type=float, default=45)
    args = parser.parse_args()
    if args.mode == 'launch':
        if args.kit_executable is None or args.watchdog_seconds <= 0:
            parser.error('launch needs --kit-executable and a positive failure watchdog')
        launch(args)
    else:
        probe(args)


if __name__ == '__main__':
    main()
