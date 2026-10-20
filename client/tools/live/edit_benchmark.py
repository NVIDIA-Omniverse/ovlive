#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive edit benchmark: ovlive against the omni-spectree/Omniverse Client Library baseline.

Characterization only, never a qualification or CI gate. Each case starts a fresh reference graph
service, a latency-injecting TCP proxy in front of it, a neutral peer reader, an optional Omniverse
Client Library peer reader and one writer, all through the proxy, then measures the time from a host
edit to its visibility at the peer, the publish throughput and the backlog. See the testing guide's
"Interactive edit benchmark" section.

Run it with the reference-service interpreter (grpcio, grpcio-tools, protobuf):

    <reference python> tools/live/edit_benchmark.py --python-extension <_live*.so> \
        --client-library <omni_client_library package> --seed 42
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import datetime
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
CLIENTS = ROOT / 'tools/live/edit_benchmark_clients.py'
REFERENCE = ROOT / 'tests/reference_service'
SESSION, LAYER, TOKEN, PRINCIPAL = 'benchmark', 'root.live', 'benchmark-token', 'benchmark-writer'
CAPABILITIES = '/nvidia.omniverse.storage.capabilities.v1alpha.CapabilitiesService/'
SCHEMA_VERSION = 1
GRAPH_RETAINED_UPDATES = 64
# The acceptance bar's baseline: the last release that keeps several UpdateLayer calls in flight per object.
BAR_CLIENT_LIBRARY_VERSION = '2.74.0'


@dataclass(frozen=True)
class Scenario:
    name: str
    prims: int
    attributes: int
    rate_hz: float  # 0 authors the turns back to back.
    turns: int
    description: str
    batch: bool = False  # Every turn is authored inside one frame, with no host jobs run between turns.


def scenarios(duration_s, burst_turns, rate_hz=60.0):
    sustained = round(rate_hz * duration_s)
    return {
        'single': Scenario('single', 1, 1, rate_hz, sustained, f'{rate_hz:g} Hz, one attribute per turn, {duration_s:g} s'),
        'multi': Scenario('multi', 50, 3, rate_hz, sustained,
                          f'{rate_hz:g} Hz, 50 prims x 3 attributes per turn, {duration_s:g} s'),
        'burst': Scenario('burst', 1, 1, 0.0, burst_turns, f'{burst_turns} one-attribute turns back to back'),
        'multi-burst': Scenario('multi-burst', 50, 3, 0.0, burst_turns,
                                f'{burst_turns} turns of 50 prims x 3 attributes back to back'),
        'batch-in-one-frame': Scenario('batch-in-one-frame', 50, 3, 0.0, burst_turns,
                                       f'{burst_turns} change blocks of 50 prims x 3 attributes in one frame', True),
    }


def layout(prims, attributes):
    """Seed graph: prims /P<i> under the root, each with double attributes a<j> whose default is -1."""
    paths, sections, nodes = [], [], []
    for prim in range(prims):
        prim_section = 1000 + prim * 16
        nodes.append(('prim', prim_section, 1, f'P{prim}'))
        for attribute in range(attributes):
            section = prim_section + 1 + attribute
            nodes.append(('attribute', section, prim_section, f'a{attribute}'))
            paths.append(f'/P{prim}.a{attribute}')
            sections.append(section)
    return nodes, paths, sections


def seed_payload(nodes):
    from scene_graph_wire import Command, Delta, Field, Specifier, Token, encode
    commands, fields = [], []
    for kind, section, parent, name in nodes:
        commands.append(Command(1, section, parent=parent, name=name, spec_type=6 if kind == 'prim' else 1))
        fields += ([Field(section, 'specifier', Specifier(0))] if kind == 'prim' else
                   [Field(section, 'typeName', Token('double')), Field(section, 'default', -1.0)])
    return encode(Delta(commands=tuple(commands), fields=tuple(fields)))


def emit(kind, **fields):
    print(json.dumps(dict(type=kind, **fields)), flush=True)


def stdin_event(loop, expected):
    event = asyncio.Event()

    def read():
        for line in sys.stdin:
            if line.strip() == expected:
                break
        loop.call_soon_threadsafe(event.set)
    threading.Thread(target=read, daemon=True).start()
    return event


def update_counts(ledger):
    """UpdateLayer commits of the benchmark layer, in total and per writer, from the service's full ledger."""
    writers = {}
    for entry in ledger:
        if entry.operation == 'UpdateLayer' and entry.session == SESSION and entry.layer == LAYER:
            counts = writers.setdefault(entry.writer or '', dict(commits=0, bytes=0))
            counts['commits'] += 1
            counts['bytes'] += len(entry.payload or b'')
    return dict(commits=sum(w['commits'] for w in writers.values()), writers=list(writers.values()))


def field(number, data):
    """One length-delimited protobuf field; the capability messages hold only strings and messages."""
    length, prefix = len(data), bytearray()
    while True:
        prefix.append((length & 0x7f) | (0x80 if length > 0x7f else 0))
        length >>= 7
        if not length:
            break
    return bytes([number << 3 | 2]) + bytes(prefix) + data


# --- service ---------------------------------------------------------------------------------------

async def serve(args):
    sys.path[:0] = [args.generated, str(REFERENCE)]
    import grpc
    from graph_service import GraphServer, GraphState
    from service import Principal, ReferenceServer, Session, pb, rpc

    class Capabilities(grpc.GenericRpcHandler):
        """The storage capability queries Omniverse Client Library makes before it uses a direct endpoint."""
        address = ''

        def service(self, details):
            name = details.method[len(CAPABILITIES):] if details.method.startswith(CAPABILITIES) else None
            if name == 'ListServices':
                body = b''.join(field(1, field(1, service.encode()) + field(2, b'v1alpha'))
                                for service in ('fileobject', 'filefolder', 'capabilities', 'live'))
            elif name == 'ListTopLevelAddresses':
                body = field(1, field(1, f'http://{self.address}/'.encode()))
            elif name == 'ListRoutes':
                body = b''
            else:
                return None
            return grpc.unary_unary_rpc_method_handler(lambda _request, _context: body)

    state = GraphState(args.seed)
    # The ledger keeps every payload; long runs need more room than the reference default. A writer that
    # coalesces a backlog sends updates up to the Live Edit service's 100 MiB upload limit.
    state.max_commits, state.max_retained_bytes, state.max_payload = 1_000_000, 4 << 30, 100 << 20
    state.tokens[TOKEN] = Principal(PRINCIPAL)
    for option in args.service_option:
        name, _, text = option.partition('=')
        if not hasattr(state, name):
            raise SystemExit(f'the reference service has no option {name}')
        value = {'true': True, 'false': False}.get(text.lower(), text)
        setattr(state, name, int(value) if isinstance(value, str) and value.isdigit() else value)
    state.sessions[SESSION] = Session(SESSION, '/benchmark', 'benchmark', PRINCIPAL, state.clock(), {PRINCIPAL})
    nodes, _, _ = layout(args.prims, args.attributes)
    if args.service == 'opaque':
        # The opaque service never decodes updates; only the seed scene, which can exceed the graph
        # profile's per-update bounds, is encoded here.
        import scene_graph_model
        import scene_graph_wire
        scene_graph_wire.MAX_BYTES = scene_graph_wire.MAX_RECORDS = scene_graph_wire.MAX_STRING_BYTES = 1 << 30
        scene_graph_model.MAX_SECTIONS = scene_graph_model.MAX_FIELDS = 1 << 30
        scene_graph_model.MAX_WORK = scene_graph_model.MAX_HISTORY_BYTES = 1 << 40
    payload = seed_payload(nodes)
    if args.service == 'graph':
        # The graph service re-derives its retained snapshot on every commit; a short retention window
        # bounds that cost, and the benchmark's readers never resume.
        state.retained_updates = GRAPH_RETAINED_UPDATES
        server = GraphServer(state)
    else:
        # The opaque service relays each committed payload verbatim, in the Live Edit service's per-commit
        # catch-up mode, so a fresh subscriber's catch-up starts with the snapshot the layer is created from.
        import scene_graph_model as graph
        from scene_graph_snapshot import snapshot
        model = graph.Model()
        model.apply(payload, 1)
        payload = snapshot(model.state, source_format=model.source_format)
        server = ReferenceServer(state)
    # Each update's commit time on the monotonic clock the clients use splits edit-to-peer latency at the
    # service; only the opaque service commits through State.commit, so the graph service has no split.
    timeline, original_commit = [], state.commit

    def timed_commit(operation, session, **fields):
        entry = original_commit(operation, session, **fields)
        if operation == 'UpdateLayer':
            timeline.append((entry.sequence, time.monotonic()))
        return entry
    state.commit = timed_commit
    capabilities = Capabilities()
    server.server.add_generic_rpc_handlers((capabilities,))
    address = capabilities.address = await server.start()
    async with grpc.aio.insecure_channel(address) as channel:
        created = await rpc.LiveServiceStub(channel).CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=SESSION, layer_name=LAYER)),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=payload))]),
            metadata=(('authorization', f'Bearer {TOKEN}'),), timeout=10)
    emit('ready', address=address, object_id=created.object_id)
    await stdin_event(asyncio.get_running_loop(), 'stop').wait()
    # The ledger holds every commit; the graph service's layer retains only its newest updates.
    emit('done', **update_counts(state.ledger), timeline=timeline)
    await server.stop()


# --- latency proxy ---------------------------------------------------------------------------------

async def proxy(args):
    """Forward TCP with `one_way_ms` plus uniform [0, jitter_ms) added to every chunk in each direction.

    Delivery keeps byte order: a chunk is never delivered before the one read ahead of it, so jitter
    delays a chunk behind a slower predecessor instead of reordering the stream. Each connection and
    direction draws its jitter from its own generator seeded from --seed and the connection index.
    """
    upstream_host, upstream_port = args.upstream.rsplit(':', 1)
    connections = 0

    async def pump(reader, writer, rng):
        loop = asyncio.get_running_loop()
        queue, last = asyncio.Queue(), 0.0

        async def deliver():
            while True:
                due, data = await queue.get()
                if data is None:
                    break
                delay = due - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                writer.write(data)
                await writer.drain()
            writer.close()
        task = asyncio.create_task(deliver())
        try:
            while data := await reader.read(65536):
                last = max(last, loop.time() + (args.one_way_ms + rng.random() * args.jitter_ms) / 1000)
                queue.put_nowait((last, data))
        except (ConnectionError, OSError):
            pass
        queue.put_nowait((0.0, None))
        try:
            await task
        except (ConnectionError, OSError):
            pass

    async def accept(client_reader, client_writer):
        nonlocal connections
        index, connections = connections, connections + 1
        try:
            server_reader, server_writer = await asyncio.open_connection(upstream_host, int(upstream_port))
        except OSError:
            client_writer.close()
            return
        await asyncio.gather(pump(client_reader, server_writer, random.Random(f'{args.seed}:{index}:up')),
                             pump(server_reader, client_writer, random.Random(f'{args.seed}:{index}:down')))

    server = await asyncio.start_server(accept, '127.0.0.1', 0)
    emit('ready', address='127.0.0.1:%d' % server.sockets[0].getsockname()[1])
    await stdin_event(asyncio.get_running_loop(), 'stop').wait()
    server.close()
    emit('done', connections=connections)


# --- neutral reader --------------------------------------------------------------------------------

async def read(args):
    """A raw SubscribeReadLayer peer: records when each committed update arrives and its turn marker."""
    sys.path[:0] = [args.generated, str(REFERENCE)]
    import grpc
    from edit_benchmark_clients import turn_marker
    from service import pb, rpc

    received, ready = [], False
    stop = stdin_event(asyncio.get_running_loop(), 'stop')
    async with grpc.aio.insecure_channel(args.authority, options=[('grpc.max_receive_message_length', 1 << 30)]) as channel:
        call = rpc.LiveServiceStub(channel).SubscribeReadLayer(pb.SubscribeReadLayerRequest(
            session_id=SESSION, layer_name=LAYER, client_id='benchmark-neutral-reader'),
            metadata=(('authorization', f'Bearer {TOKEN}'),))

        async def consume():
            nonlocal ready
            header, data = None, b''
            async for message in call:
                if message.HasField('header'):
                    header, data = message.header, b''
                    if header.total_size == 0 and not ready:
                        ready = True  # The caught-up header ends the catch-up.
                        emit('ready')
                        continue
                else:
                    data += message.chunk.data
                if header is not None and header.total_size and len(data) == header.total_size:
                    now = time.monotonic()
                    if ready:
                        # Decoding waits until the read ends, so the reader keeps pace with any writer.
                        received.append((now, header.sequence_num, bytes(data)))
                    header = None
        task = asyncio.create_task(consume())
        await stop.wait()
        task.cancel()
    for index, (now, sequence, data) in enumerate(received):
        received[index] = (now, sequence, turn_marker(data), len(data))
        call.cancel()
    Path(args.output).write_text(json.dumps(dict(client='neutral-reader', received=received)))
    emit('done')


# --- orchestration ---------------------------------------------------------------------------------

class Child:
    def __init__(self, name, process, log, handle):
        self.name, self.process, self.log, self.handle = name, process, log, handle

    @classmethod
    async def start(cls, name, command, log, environment=None):
        # The parent copies library lines from the child's stdout through the handle the child inherits as
        # stderr, so both writers share one file position: O_APPEND on POSIX, and on Windows, whose C runtime
        # only emulates append per descriptor, the file object the inherited handle duplicates.
        handle = open(log, 'ab', buffering=0)
        try:
            process = await asyncio.create_subprocess_exec(*command, stdin=asyncio.subprocess.PIPE,
                                                           stdout=asyncio.subprocess.PIPE, stderr=handle,
                                                           env=environment, limit=1 << 24)
        except BaseException:
            handle.close()
            raise
        return cls(name, process, log, handle)

    async def receive(self, timeout):
        """Next JSON record on stdout; library log lines that share stdout are copied to the log."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f'{self.name} sent no JSON record within {timeout:g} s; see {self.log}')
            line = await asyncio.wait_for(self.process.stdout.readline(), remaining)
            if not line:
                raise RuntimeError(f'{self.name} exited ({await self.process.wait()}); see {self.log}')
            if line.startswith(b'{'):
                return json.loads(line)
            self.handle.write(line)

    async def expect(self, kind, timeout):
        record = await self.receive(timeout)
        if record.get('type') != kind:
            raise RuntimeError(f'{self.name}: expected {kind}, got {record}')
        return record

    def send(self, line):
        self.process.stdin.write(line.encode() + b'\n')

    async def close(self):
        if self.process.returncode is None:
            self.process.kill()
        await self.process.wait()
        self.handle.close()


def read_manifest(extension):
    return json.loads(Path(str(extension) + '.manifest.json').read_text())


def extension_manifest(extension):
    """The SDK identity manifest the build writes beside the Python extension, checked against what loads.

    The extension, the plugin artifacts and the SDK trees the manifest names must hash as recorded. Sources
    are not compared with this checkout, since a benchmark can measure another revision's build.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.live.python_identity import digest
    from tools.live.wire_identity import tree_hash
    manifest = read_manifest(extension)
    if manifest.get('schema_version') != 1 or manifest.get('boundary') != 'python_adapter':
        raise ValueError(f'{extension}: not a Python binding identity manifest')
    if manifest.get('artifact_sha256') != digest(extension):
        raise ValueError(f'{extension}: extension differs from its identity manifest; rebuild required')
    plugin = manifest['plugin_identity']
    if plugin.get('artifact_sha256') != digest(manifest['plugin_driver']):
        raise ValueError(f"{manifest['plugin_driver']}: plugin differs from the identity manifest; rebuild required")
    for entry in plugin.get('dependencies', []):
        if digest(entry['path']) != entry['sha256']:
            raise ValueError(f"{entry['path']}: plugin {entry['role']} differs from the identity manifest; "
                             'rebuild required')
    for entry in plugin.get('dependency_trees', []):
        if tree_hash(Path(entry['path'])) != entry['sha256']:
            raise ValueError(f"{entry['path']}: SDK {entry['role']} differs from the identity manifest; "
                             'rebuild required')
    return manifest


# The USD runtime configuration both USD hosts clear, so the invoking shell configures neither.
USD_CONFIGURATION = dict(PXR_PLUGINPATH_NAME='', PXR_AR_DEFAULT_SEARCH_PATH='', AR_DEFAULT_RESOLVER='')


def extension_environment(extension):
    """The environment the python-reference lane gives the packaged Python bindings."""
    trees = read_manifest(extension)['plugin_identity']['dependency_trees']
    libraries = [e['path'] for e in trees if e['role'] in {'usd_lib', 'python_lib'}]
    usd_python = [str(Path(e['path']) / 'python') for e in trees if e['role'] == 'usd_lib']
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('PXR_', 'PYTHON'))}
    environment.update(LD_LIBRARY_PATH=os.pathsep.join(libraries),
                       PYTHONPATH=os.pathsep.join([str(extension.parents[3]), *usd_python]),
                       **USD_CONFIGURATION, PYTHONDONTWRITEBYTECODE='1')
    return environment


def client_library_environment(package):
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'OMNI_'))}
    environment.update(PYTHONPATH=str(package / 'release/bindings-python'), PYTHONDONTWRITEBYTECODE='1')
    return environment


def baseline_usd_environment(package, extension):
    """The client-library writer's environment: the baseline package with the ovlive build's USD SDK,
    whose runtime is configured as the ovlive writer's is."""
    usd = extension_environment(extension)
    environment = {k: v for k, v in client_library_environment(package).items() if not k.startswith('PXR_')}
    usd_python = usd['PYTHONPATH'].split(os.pathsep)[1:]  # Without the ovlive package.
    environment.update(LD_LIBRARY_PATH=usd['LD_LIBRARY_PATH'], **USD_CONFIGURATION,
                       PYTHONPATH=os.pathsep.join([environment['PYTHONPATH'], *usd_python]))
    return environment


def client_library_version(package):
    """The version and build string the Omniverse Client Library package reports."""
    return subprocess.run([sys.executable, '-c', 'import omni.client; print(omni.client.get_version())'],
                          env=client_library_environment(package), check=True, capture_output=True, text=True,
                          timeout=60).stdout.strip()


def identities(python_extension, manifest, client_library, version):
    """What a run measured: the ovlive extension and its SDK identity manifest, which also names the USD
    libraries the client-library writer authors with, and the baseline package and its reported version."""
    return dict(ovlive=None if python_extension is None else dict(python_extension=str(python_extension),
                                                                  manifest=manifest),
                client_library=None if client_library is None else dict(package=str(client_library), version=version))


def bar_baseline(version):
    """Whether a reported client library version, with any build suffix, is the bar's baseline release."""
    return version is not None and (version == BAR_CLIENT_LIBRARY_VERSION
                                    or version.startswith((BAR_CLIENT_LIBRARY_VERSION + '-',
                                                           BAR_CLIENT_LIBRARY_VERSION + '+')))


def check_client_library_versions(report):
    """Every baseline writer reports the version recorded in the report's identities."""
    expected = (report['identities']['client_library'] or {}).get('version')
    for case in report['cases']:
        reported = case['metrics']['writer'].get('version')
        if reported is not None and reported != expected:
            raise RuntimeError(f"{case['scenario']} rtt {case['rtt_ms']} {case['client']}: client library "
                               f"{reported} differs from the recorded {expected}")


async def run_case(args, scenario, rtt_ms, client, directory):
    directory.mkdir(parents=True)
    python, children = sys.executable, []
    common = ['--generated', str(args.generated), '--seed', str(args.seed)]
    try:
        options = [f'--service-option={option}' for option in args.service_option]
        service = await Child.start('service', [python, __file__, 'serve', *common, '--service', args.service, *options,
                                                '--prims', str(max(scenario.prims, args.scene_prims)),
                                                '--attributes', str(scenario.attributes)], directory / 'service.log')
        children.append(service)
        ready = await service.expect('ready', 60)
        proxy_child = await Child.start('proxy', [python, __file__, 'proxy', '--upstream', ready['address'],
                                                  '--one-way-ms', str(rtt_ms / 2), '--jitter-ms', str(args.jitter_ms),
                                                  '--seed', str(args.seed)], directory / 'proxy.log')
        children.append(proxy_child)
        authority = (await proxy_child.expect('ready', 30))['address']
        readers = [await Child.start('neutral-reader', [python, __file__, 'read', *common, '--authority', authority,
                                                         '--output', str(directory / 'neutral-reader.json')],
                                     directory / 'neutral-reader.log')]
        connection = ['--authority', authority, '--token', TOKEN, '--session', SESSION, '--layer', LAYER,
                      *(['--log-level', args.client_library_log_level] if args.client_library_log_level else [])]
        if args.client_library and not args.no_client_library_reader:
            readers.append(await Child.start('client-library-reader', [
                python, str(CLIENTS), 'client-library-reader', *connection,
                '--output', str(directory / 'client-library-reader.json')],
                directory / 'client-library-reader.log', client_library_environment(args.client_library)))
        children += readers
        for reader in readers:
            await reader.expect('ready', 60)
        _, paths, sections = layout(scenario.prims, scenario.attributes)
        workload = ['--turns', str(scenario.turns), '--rate-hz', str(scenario.rate_hz),
                    '--output', str(directory / 'writer.json'), '--drain-timeout-s', str(args.drain_timeout_s),
                    *(['--batch'] if scenario.batch else [])]
        # A wrapper such as a sampling profiler runs the writer; `{dir}` names the case directory.
        wrapper = [part.replace('{dir}', str(directory)) for part in shlex.split(args.writer_wrapper or '')]
        if client == 'ovlive':
            writer = await Child.start('writer', [
                *wrapper, python, str(CLIENTS), 'ovlive-writer', *connection, *workload,
                '--package', str(args.python_extension.parents[4]), '--attributes', ','.join(paths),
                '--frame-wait-ms', str(args.frame_wait_ms)],
                directory / 'writer.log', extension_environment(args.python_extension))
        else:
            # client-library authors USD as omni-spectree's host does; client-library-record only records values.
            model = 'record' if client == 'client-library-record' else 'usd'
            environment = (baseline_usd_environment(args.client_library, args.python_extension) if model == 'usd'
                           else client_library_environment(args.client_library))
            writer = await Child.start('writer', [
                *wrapper, python, str(CLIENTS), 'client-library-writer', *connection, *workload,
                '--sections', ','.join(map(str, sections)), '--attributes', ','.join(paths),
                # The baseline host's layer holds the same scene as the live layer.
                '--scene-prims', str(max(scenario.prims, args.scene_prims)),
                '--scene-attributes', str(scenario.attributes),
                '--baseline-model', model, '--object-id', str(ready['object_id'])],
                directory / 'writer.log', environment)
        children.append(writer)
        await writer.expect('ready', 60)
        writer.send('go')
        await writer.expect('done', scenario.turns / (scenario.rate_hz or 100) + args.drain_timeout_s + 60)
        # Every write is acknowledged; give the last update time to reach the peers.
        await asyncio.sleep(1.0 + 4 * rtt_ms / 1000)
        for reader in readers:
            reader.send('stop')
        for reader in readers:
            await reader.expect('done', 600)  # Readers decode what they received after the read ends.
        writer.send('exit')
        # The writer ends while its service is reachable, so a wrapper sees a clean exit.
        try:
            await asyncio.wait_for(writer.process.wait(), 60)
        except asyncio.TimeoutError:
            pass
        service.send('stop')
        commits = await service.expect('done', 30)
        proxy_child.send('stop')
        await proxy_child.expect('done', 30)
    finally:
        for child in children:
            await child.close()
    result = dict(writer=json.loads((directory / 'writer.json').read_text()),
                  neutral=json.loads((directory / 'neutral-reader.json').read_text()), service=commits)
    jitter_reader = directory / 'client-library-reader.json'
    if jitter_reader.exists():
        result['jitter'] = json.loads(jitter_reader.read_text())
    return result


def percentile(values, fraction):
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] if ordered else None


def visibility(edits, received):
    """Time each turn became visible: the first update whose marker is at or past the turn."""
    arrivals = sorted((t, marker) for t, _sequence, marker, _size in received if marker is not None)
    visible, index, highest = {}, 0, -math.inf
    for turn, _edit in edits:
        while highest < turn and index < len(arrivals):
            highest = max(highest, arrivals[index][1])
            arrival = arrivals[index][0]
            index += 1
        if highest >= turn:
            visible[turn] = arrival
    return visible


def peer_metrics(edits, received):
    if not edits:
        return dict(turns=0, visible_turns=0, latency_ms=dict(p50=None, p95=None, p99=None, max=None),
                    throughput_turns_per_s=None, authored_turns_per_s=None, drain_ms=None, max_backlog_turns=0,
                    updates=0, turns_per_update=None)
    visible = visibility(edits, received)
    latencies = [(visible[turn] - edit) * 1000 for turn, edit in edits if turn in visible]
    first_edit, last_edit = edits[0][1], edits[-1][1]
    times = sorted(visible.values())
    backlog, cursor = 0, 0
    for authored, (_turn, edit) in enumerate(edits, 1):
        while cursor < len(times) and times[cursor] <= edit:
            cursor += 1
        backlog = max(backlog, authored - cursor)
    updates = sum(1 for t, _s, marker, _n in received if marker is not None and t >= first_edit)
    complete = len(visible) == len(edits)
    return dict(turns=len(edits), visible_turns=len(visible),
                latency_ms=dict(p50=percentile(latencies, .5), p95=percentile(latencies, .95),
                                p99=percentile(latencies, .99), max=max(latencies, default=None)),
                throughput_turns_per_s=len(edits) / (times[-1] - first_edit) if complete and times[-1] > first_edit else None,
                authored_turns_per_s=(len(edits) - 1) / (last_edit - first_edit) if last_edit > first_edit else None,
                drain_ms=(times[-1] - last_edit) * 1000 if complete else None,
                max_backlog_turns=backlog, updates=updates,
                turns_per_update=len(edits) / updates if updates else None)


OVLIVE_FAILED_STATE = 7  # omni.usd.live.State.FAILED


def writer_metrics(writer):
    rejected = writer.get('rejected', [])
    accepted = dict(accepted_turns=len(writer['edits']), rejected_turns=len(rejected),
                    first_rejected_turn=rejected[0][0] if rejected else None,
                    rejection=rejected[0][2:] if rejected else None)
    if writer['client'] == 'ovlive':
        # A failed write fails the layer, which ends its drain; a drain that ends otherwise timed out.
        final = writer['final']
        failures = [f"layer FAILED, result {final['result']}"] if final['state'] == OVLIVE_FAILED_STATE else []
        return dict(**accepted, frame_timeouts=writer.get('frame_timeouts'), failures=failures,
                    drained=writer['drained'], drain_timed_out=not writer['drained'] and not failures,
                    peak_pending_write_count=writer['peak']['pending_write_count'],
                    peak_buffered_edit_count=writer['peak']['buffered_edit_count'],
                    blocked_turns=writer['blocked_turns'], final=writer['final'],
                    delivery_status=writer['delivery_status'], metrics=writer['metrics'])
    # Every baseline write settles in its callback, failed or not, so an undrained baseline timed out.
    return dict(**accepted, drained=writer['drained'], drain_timed_out=not writer['drained'],
                window=writer['window'],
                peak_in_flight=writer['peak']['in_flight'],
                peak_buffered_turns=writer['peak']['buffered_turns'], updates_sent=len(writer['updates']),
                failures=writer['failures'], version=writer['version'])


def split_at_service(edits, received, timeline):
    """Edit-to-commit and commit-to-peer times of each turn, through the update that made it visible."""
    committed = dict((sequence, at) for sequence, at in timeline)
    arrivals = sorted((t, sequence, marker) for t, sequence, marker, _size in received if marker is not None)
    upload, fanout, index, highest, sequence, arrival = [], [], 0, -math.inf, None, None
    for turn, edit in edits:
        while highest < turn and index < len(arrivals):
            arrival, sequence, marker = arrivals[index]
            highest = max(highest, marker)
            index += 1
        if highest >= turn and sequence in committed:
            upload.append((committed[sequence] - edit) * 1000)
            fanout.append((arrival - committed[sequence]) * 1000)
    return {name: dict(p50=percentile(values, .5), p99=percentile(values, .99))
            for name, values in (('edit_to_commit_ms', upload), ('commit_to_peer_ms', fanout))}


def delivery(metrics):
    """Whether a case delivered every turn its writer authored.

    A delivery is complete when the writer accepted every turn (none refused), the peer saw every accepted
    turn, no write failed and the writer drained. Latency and throughput describe only the turns the peer
    saw, so they compare only between complete deliveries. An ovlive layer that fails counts one failed
    write, the one that failed it; a writer that stops draining without a failure timed out.
    """
    peer, writer = metrics['peer'], metrics['writer']
    refused, failed = writer.get('rejected_turns', 0), len(writer.get('failures') or [])
    record = dict(authored_turns=peer['turns'] + refused, visible_turns=peer['visible_turns'], refused_turns=refused,
                  failed_writes=failed, drained=writer.get('drained') is True,
                  drain_timed_out=writer.get('drain_timed_out') is True)
    record['complete'] = (peer['turns'] > 0 and peer['visible_turns'] == peer['turns'] and not refused
                          and not failed and record['drained'])
    return record


def analyze(raw):
    edits = raw['writer']['edits']
    service = {k: v for k, v in raw['service'].items() if k != 'timeline'}
    result = dict(peer=peer_metrics(edits, raw['neutral']['received']), writer=writer_metrics(raw['writer']),
                  service=service)
    result['delivery'] = delivery(result)
    if raw['service'].get('timeline'):
        result['peer']['split'] = split_at_service(edits, raw['neutral']['received'], raw['service']['timeline'])
    if 'jitter' in raw:
        result['peer_jitter_reduction'] = peer_metrics(edits, raw['jitter']['received'])
    return result


def fmt(value, digits=1):
    return '-' if value is None else f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def identity_line(measured):
    ovlive, baseline = measured['ovlive'], measured['client_library']
    parts = [f"ovlive extension sha256 {ovlive['manifest'].get('artifact_sha256')}" if ovlive
             else 'no ovlive extension',
             f"Omniverse Client Library {baseline['version']}" if baseline else 'no Omniverse Client Library']
    manifest = '; results.json holds the extension\'s SDK identity manifest' if ovlive else ''
    return 'Measured: ' + ', '.join(parts) + manifest + '.'


def summary(report):
    lines = ['# Interactive edit benchmark', '',
             f"Seed {report['seed']}, {report['service']} reference service, proxy jitter {report['jitter_ms']} ms, labels: "
             + ', '.join(f'{k}={v}' for k, v in report['labels'].items()) + '.', '',
             identity_line(report['identities']), '',
             'Latency is host edit to peer visibility at a raw SubscribeReadLayer reader; the jitter column is the '
             'same p50/p99 at an Omniverse Client Library reader with its default jitter reduction. The client-library '
             'writer coalesces edits per 60 Hz tick (per loop iteration in the burst) and keeps at most 16 updates '
             'in flight, as omni-spectree\'s default live layer format does.', '']
    header = ('| Scenario | RTT ms | Client | p50 ms | p95 ms | p99 ms | Edit-to-commit / commit-to-peer p50 ms | '
              'Jitter-reader p50/p99 ms | Published turns/s | Authored turns/s | Max backlog turns | Updates | '
              'Turns/update | Drain ms | Writer peak |')
    lines += [header, '|' + ' --- |' * (header.count('|') - 1)]
    for case in report['cases']:
        peer, writer = case['metrics']['peer'], case['metrics']['writer']
        jitter = case['metrics'].get('peer_jitter_reduction')
        jitter_text = f"{fmt(jitter['latency_ms']['p50'])}/{fmt(jitter['latency_ms']['p99'])}" if jitter else '-'
        split = peer.get('split')
        split_text = (f"{fmt(split['edit_to_commit_ms']['p50'])} / {fmt(split['commit_to_peer_ms']['p50'])}"
                      if split else '-')
        writes = writer.get('metrics', {}).get('layer_writes') if case['client'] == 'ovlive' else None
        window = (f", in flight {writes.get('in_flight_high_water')}, window {writes.get('window_high_water')}, "
                  f"coalesced turns {writes.get('coalesced_turns')}, max turns/update {writes.get('max_turns_per_update')}"
                  if isinstance(writes, dict) else '')
        peak = (f"pending {writer['peak_pending_write_count']}, buffered {writer['peak_buffered_edit_count']}, "
                f"blocked turns {writer['blocked_turns']}{window}" if case['client'] == 'ovlive' else
                f"in flight {writer['peak_in_flight']}/{writer['window']}, buffered turns {writer['peak_buffered_turns']}")
        if writer.get('rejected_turns'):
            peak += (f", accepted {writer['accepted_turns']} turns, refused {writer['rejected_turns']} "
                     f"from turn {writer['first_rejected_turn']}")
        delivered = delivery(case['metrics'])
        if not delivered['complete']:
            drain = ('drained' if delivered['drained'] else 'drain timed out' if delivered['drain_timed_out']
                     else 'not drained')
            peak += (f", incomplete: peer saw {delivered['visible_turns']}/{delivered['authored_turns']} turns, "
                     f"{delivered['failed_writes']} failed writes, {drain}")
        lines.append(f"| {case['scenario']} | {case['rtt_ms']} | {case['client']} | {fmt(peer['latency_ms']['p50'])} | "
                     f"{fmt(peer['latency_ms']['p95'])} | {fmt(peer['latency_ms']['p99'])} | {split_text} | {jitter_text} | "
                     f"{fmt(peer['throughput_turns_per_s'])} | {fmt(peer['authored_turns_per_s'])} | "
                     f"{peer['max_backlog_turns']} | {peer['updates']} | {fmt(peer['turns_per_update'], 2)} | "
                     f"{fmt(peer['drain_ms'])} | {peak} |")
    verdicts = comparisons(report['cases'])
    if verdicts:
        lines += ['', '## ovlive against the baseline', '',
                  'Each cell is ovlive/baseline. The bar is complete delivery on both sides, p50 and p99 for every '
                  'scenario and published throughput for the sustained scenarios; a cell in the bar where ovlive is '
                  'worse is marked (worse). Turns delivered counts the turns the peer saw of the turns authored; '
                  'a side that refused, lost or failed a turn, or never drained, is marked (incomplete) and the row '
                  'misses the bar, because latency covers only the turns a peer saw. An incomplete baseline is no '
                  'bar to meet, so its row needs a rerun. p95, backlog and burst throughput are marked (info) and '
                  f'never decide the bar. The bar is against Omniverse Client Library {BAR_CLIENT_LIBRARY_VERSION}; '
                  'a row whose baseline reports another version is a characterization and misses the bar.', '',
                  '| Scenario | RTT ms | p50 | p95 | p99 | Published turns/s | Max backlog | Turns delivered '
                  '| Meets the bar |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for v in verdicts:
            cells = []
            for column, (o, b, ok, in_bar) in enumerate(v['measurements']):
                incomplete = column == len(v['measurements']) - 1
                note = ' (info)' if not in_bar else '' if ok else ' (incomplete)' if incomplete else ' (worse)'
                cells.append(f'{fmt(o)}/{fmt(b)}{note}')
            reasons = ([f"{', '.join(v['incomplete'])} incomplete"] if v['incomplete'] else []) + (
                [] if v['qualified'] else [f"baseline {v['baseline_version']}, not {BAR_CLIENT_LIBRARY_VERSION}"])
            verdict = 'yes' if v['meets'] else 'no' + (f" ({'; '.join(reasons)})" if reasons else '')
            lines.append(f"| {v['scenario']} | {v['rtt_ms']} | {' | '.join(cells)} | {verdict} |")
    return '\n'.join(lines) + '\n'


# Scenarios whose turns come back to back. The Kit-frame writer's per-frame flush caps their published
# throughput by construction, so the bar compares their latency only.
BURST_SCENARIOS = {'burst', 'multi-burst', 'batch-in-one-frame'}


def comparisons(cases):
    """ovlive against the client-library baseline per scenario and RTT.

    Each measurement is (ovlive, baseline, at least as good, in the bar). The bar is complete delivery on
    both sides, p50 and p99 for every scenario and published throughput for the sustained ones; p95,
    backlog and burst throughput are informational. Latency covers only the turns a peer saw, so a side
    that refused, lost or failed a turn, or never drained, makes the row miss the bar: an incomplete
    ovlive run cannot meet it, and an incomplete baseline is no bar to meet, so the row needs a rerun.
    `incomplete` names the incomplete sides. The bar is against client library 2.74.0, so a row whose
    baseline reports another version is not `qualified` and misses the bar whatever it measured.
    """
    index = {(c['scenario'], c['rtt_ms'], c['client']): c['metrics'] for c in cases}
    verdicts = []
    for (scenario, rtt_ms, client), ovlive in index.items():
        baseline = index.get((scenario, rtt_ms, 'client-library'))
        if client != 'ovlive' or baseline is None:
            continue
        mine, theirs = ovlive['peer'], baseline['peer']
        delivered = dict(ovlive=delivery(ovlive), baseline=delivery(baseline))
        incomplete = [side for side, record in delivered.items() if not record['complete']]
        measurements = []
        for key in ('p50', 'p95', 'p99'):
            o, b = mine['latency_ms'][key], theirs['latency_ms'][key]
            measurements.append((o, b, o is not None and b is not None and o <= b, key != 'p95'))
        o, b = mine['throughput_turns_per_s'], theirs['throughput_turns_per_s']
        measurements.append((o, b, o is not None and (b is None or o >= b), scenario not in BURST_SCENARIOS))
        o, b = mine['max_backlog_turns'], theirs['max_backlog_turns']
        measurements.append((o, b, o <= b, False))
        o, b = (f"{r['visible_turns']} of {r['authored_turns']}" for r in delivered.values())
        measurements.append((o, b, not incomplete, True))
        version = baseline['writer'].get('version')
        qualified = bar_baseline(version)
        verdicts.append(dict(scenario=scenario, rtt_ms=rtt_ms, measurements=measurements, delivery=delivered,
                             incomplete=incomplete, baseline_version=version, qualified=qualified,
                             meets=qualified and all(ok for _o, _b, ok, in_bar in measurements if in_bar)))
    return verdicts


def generate_stubs(destination):
    destination.mkdir(parents=True, exist_ok=True)
    proto_root = ROOT / 'source/live/protos'
    subprocess.run([sys.executable, '-m', 'grpc_tools.protoc', f'--proto_path={proto_root}',
                    f'--python_out={destination}', f'--grpc_python_out={destination}',
                    str(proto_root / 'nvidia/omniverse/live/v1alpha/live.proto')], check=True, timeout=60)


async def run(args):
    selected = scenarios(args.duration_s, args.burst_turns, args.rate_hz)
    names = args.scenarios.split(',')
    clients = args.clients.split(',')
    if set(names) - set(selected) or set(clients) - {'ovlive', 'client-library', 'client-library-record'}:
        raise SystemExit('unknown scenario or client')
    if {'ovlive', 'client-library'} & set(clients) and not args.python_extension:
        # The client-library baseline authors USD with the SDK that build names.
        raise SystemExit('--python-extension is required for the ovlive and client-library clients')
    if {'client-library', 'client-library-record'} & set(clients) and not args.client_library:
        raise SystemExit('--client-library is required for the client-library clients')
    args.python_extension = args.python_extension and args.python_extension.resolve()
    args.client_library = args.client_library and args.client_library.resolve()
    output = (args.output and args.output.resolve()) or ROOT.parent / '.cache/edit-benchmark' / datetime.datetime.now().strftime('run-%Y%m%d-%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    args.generated = output / 'generated'
    generate_stubs(args.generated)
    labels = dict(label.split('=', 1) for label in args.label)
    measured = identities(args.python_extension,
                          args.python_extension and extension_manifest(args.python_extension), args.client_library,
                          args.client_library and client_library_version(args.client_library))
    report = dict(schema_version=SCHEMA_VERSION, qualification='characterization_only', seed=args.seed,
                  service=args.service, service_options=args.service_option, jitter_ms=args.jitter_ms, labels=labels,
                  identities=measured, scenarios={n: selected[n].description for n in names},
                  scene_prims=args.scene_prims, cases=[])
    for name in names:
        for rtt_ms in [int(v) for v in args.rtt_ms.split(',')]:
            for client in clients:
                directory = output / f'{name}-rtt{rtt_ms}-{client}'
                print(f'{directory.name} ...', file=sys.stderr, flush=True)
                raw = await run_case(args, selected[name], rtt_ms, client, directory)
                metrics = analyze(raw)
                (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
                report['cases'].append(dict(scenario=name, rtt_ms=rtt_ms, client=client, metrics=metrics))
                peer, delivered = metrics['peer'], metrics['delivery']
                print(f"  p50 {fmt(peer['latency_ms']['p50'])} ms, p99 {fmt(peer['latency_ms']['p99'])} ms, "
                      f"{fmt(peer['throughput_turns_per_s'])} turns/s, backlog {peer['max_backlog_turns']}, "
                      f"delivered {delivered['visible_turns']}/{delivered['authored_turns']}"
                      + ('' if delivered['complete'] else ' (incomplete)'), file=sys.stderr, flush=True)
    report['comparisons'] = comparisons(report['cases'])
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'summary.md').write_text(summary(report))
    print(output / 'summary.md')
    check_client_library_versions(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command')
    run_parser = sub.add_parser('run', help='run the benchmark matrix (the default)')
    for p in (parser, run_parser):
        p.add_argument('--python-extension', type=Path, help='built omni/usd/live/_live*.so of the ovlive under test')
        p.add_argument('--client-library', type=Path,
                       help='Omniverse Client Library package root holding release/bindings-python (the baseline)')
        p.add_argument('--clients', default='ovlive,client-library')
        p.add_argument('--scenarios', default='single,multi,burst')
        p.add_argument('--rtt-ms', default='20,50,100', help='added round trips; each direction gets half')
        p.add_argument('--jitter-ms', type=float, default=2.0, help='uniform added delay per chunk and direction')
        p.add_argument('--duration-s', type=float, default=10.0, help='length of the sustained scenarios')
        p.add_argument('--burst-turns', type=int, default=1000)
        p.add_argument('--rate-hz', type=float, default=60.0, help='turn rate of the sustained scenarios')
        p.add_argument('--frame-wait-ms', type=float, default=2.0,
                       help='longest wait at an ovlive frame end for the flush job the turn requested')
        p.add_argument('--scene-prims', type=int, default=0,
                       help='prims in the layer; turns edit only the scenario\'s first prims')
        p.add_argument('--writer-wrapper', help='command prefix for the writer, such as a sampling profiler; '
                                                '{dir} expands to the case directory')
        p.add_argument('--service', choices=('opaque', 'graph'), default='opaque',
                       help='opaque relays payloads verbatim; graph validates every commit at much higher cost')
        p.add_argument('--service-option', action='append', default=[],
                       help='NAME=VALUE set on the reference service state, such as chained_writes=false')
        p.add_argument('--drain-timeout-s', type=float, default=120.0)
        p.add_argument('--client-library-log-level', help='Omniverse Client Library log level, such as debug')
        p.add_argument('--no-client-library-reader', action='store_true',
                       help='skip the jitter-reduction peer reader')
        p.add_argument('--label', action='append', default=[], help='KEY=VALUE recorded in the report')
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--output', type=Path, help='fresh results directory (default .cache/edit-benchmark/run-*)')
    serve_parser = sub.add_parser('serve')
    read_parser = sub.add_parser('read')
    proxy_parser = sub.add_parser('proxy')
    for p in (serve_parser, read_parser):
        p.add_argument('--generated', required=True)
        p.add_argument('--seed', type=int, required=True)
    serve_parser.add_argument('--service', choices=('opaque', 'graph'), required=True)
    serve_parser.add_argument('--service-option', action='append', default=[])
    serve_parser.add_argument('--prims', type=int, required=True)
    serve_parser.add_argument('--attributes', type=int, required=True)
    read_parser.add_argument('--authority', required=True)
    read_parser.add_argument('--output', required=True)
    proxy_parser.add_argument('--upstream', required=True)
    proxy_parser.add_argument('--one-way-ms', type=float, required=True)
    proxy_parser.add_argument('--jitter-ms', type=float, required=True)
    proxy_parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    if args.command in ('serve', 'read'):
        sys.path[:0] = [str(REFERENCE)]
    asyncio.run({'serve': serve, 'read': read, 'proxy': proxy}.get(args.command, run)(args))


if __name__ == '__main__':
    main()
