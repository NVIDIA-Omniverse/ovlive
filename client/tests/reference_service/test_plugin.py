# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh public-header-only consumers of the packaged production shared plugin."""
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import unittest
from urllib.parse import quote

import grpc
from service import State, Principal, ReferenceServer, pb, rpc
from scene_wire import Model, decode_update, empty_snapshot, update
from graph_service import GraphState, GraphServer
from scene_graph_wire import decode as graph_decode
import scene_graph_model as graph_model
import test_scene as scene_support
import tls_fixture
from writer_history import (generate as writer_plan, account as account_writers,
                            load as load_writer_plan, validate as validate_writer_plan, WriterMismatch)
from writer_reduction import reduce_history as reduce_writers, publish as publish_writer
from tools.live import writer_campaign as campaign

WRITER = (("authorization", "Bearer synthetic-writer"),)


class CorruptWriterService(scene_support.CompactSceneService):
    """Named test-only service defect: corrupt one edit before committing it.

    Real plugin processes still admit, issue and acknowledge the original edit.
    This exercises semantic failure reduction; no production fault switch exists.
    """
    async def chunks(self, stream, context):
        header, payload = await super().chunks(stream, context)
        if not isinstance(header, pb.UpdateLayerHeader):
            return header, payload
        fields = decode_update(payload)
        if 'owned_0_2' in fields:
            fields['owned_0_2'] = -1
            payload = update(fields)
        return header, payload


async def writer_tasks(*operations):
    """Own every trial command until success or sibling cancellation settles."""
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(operation) for operation in operations]
    return [task.result() for task in tasks]


async def client_update_attempts(driver):
    """UpdateLayer attempts one client process issued, from its public diagnostics."""
    metrics = (await driver.command('diagnostics'))['snapshot']['metrics']
    return sum(item['count'] for item in metrics['rpc_attempts_by_operation'] if item['operation'] == 4)


def account_direct_prims(authored, commits):
    """Verify each plain-USD prim edit's commit, then project lineage by structural_history's rules.

    authored: change blocks in authoring order, each {edit, writer, kind, prim,
    value} with kind define, set or remove. commits: the service ledger after
    setup. Every change block is exactly one commit, in authoring order, by its
    writer's one stable wire identity. Each commit is decoded and must carry its
    edit alone: a define creates the root prim and its value attribute with that
    default, a set carries only that attribute's default, and a remove deletes
    only that prim. An edit is reflected, or superseded when a later commit
    overwrote its value or removed its prim, including a value committed after
    its prim was removed, which the service keeps as orphan data. A define is
    superseded only by a removal. Returns the projected prims and the lineage.
    """
    if len(commits) != len(authored):
        raise AssertionError(f'{len(authored)} change blocks but {len(commits)} commits')
    identities, alive, values, members, owner, removed, lineage = {}, set(), {}, {}, {}, {}, {}
    sections = {}
    for edit, commit in zip(authored, commits):
        if commit.writer == 'independent-peer' or identities.setdefault(edit['writer'], commit.writer) != commit.writer:
            raise AssertionError(f"{edit['edit']} was committed by another wire identity")
        name, prim = edit['edit'], edit['prim']
        delta = graph_decode(commit.payload)
        fields = [(field.section, field.key, field.value) for field in delta.fields]
        if edit['kind'] == 'define':
            created = [command for command in delta.commands if command.tag in (1, 5)]
            node = next((command.section for command in created if command.parent == 1
                         and command.name == prim[1:] and command.spec_type == graph_model.PRIM), None)
            attribute = next((command.section for command in created if node is not None and command.parent == node
                              and command.name == 'value' and command.spec_type == graph_model.ATTRIBUTE), None)
            matched = (attribute is not None and len(created) == 2 and len(delta.commands) == 2
                       and prim not in sections and not delta.samples
                       and all(section in (node, attribute) for section, key, value in fields)
                       and [value for section, key, value in fields if key == 'default'] == [edit['value']]
                       and (attribute, 'default', edit['value']) in fields)
            if matched:
                sections[prim] = (node, attribute)
        elif edit['kind'] == 'set':
            matched = (prim in sections and not delta.commands and not delta.samples
                       and fields == [(sections[prim][1], 'default', edit['value'])])
        else:
            matched = (prim in sections and not fields and not delta.samples
                       and [(command.tag, command.section) for command in delta.commands] == [(2, sections[prim][0])])
        if not matched:
            raise AssertionError(f"{name} {edit['kind']} {prim} does not match its commit: {delta}")
        entry = lineage[name] = dict(sequence=commit.sequence, writer=edit['writer'], kind=edit['kind'], prim=prim)
        if edit['kind'] == 'define':
            alive.add(prim)
            values[prim] = edit['value']
            members[prim] = [name]
            owner.pop(prim, None)
        elif prim not in alive:
            entry['superseded_by'] = removed[prim]
        elif edit['kind'] == 'set':
            if prim in owner:
                lineage[owner[prim]]['superseded_by'] = name
            owner[prim] = name
            values[prim] = edit['value']
            members[prim].append(name)
        elif edit['kind'] == 'remove':
            for member in members.pop(prim):
                lineage[member].setdefault('superseded_by', name)
            alive.discard(prim)
            values.pop(prim)
            owner.pop(prim, None)
            removed[prim] = name
        else:
            raise AssertionError(f"unknown edit kind {edit['kind']}")
    if len(set(identities.values())) != len(identities):
        raise AssertionError('independent writers share a wire identity')
    for entry in lineage.values():
        entry['outcome'] = 'superseded' if 'superseded_by' in entry else 'reflected'
    return dict(paths=sorted(alive), values={prim: values[prim] for prim in sorted(alive)}), lineage


def rotation_writer_plan(master_seed, clients, rotations):
    """One root-field edit per writer per rotation, over a bounded field set.

    Even steps cycle through four owned fields and odd steps pick one of three
    shared fields, so the scene stays small however many rotations run. Every
    value is unique, so a committed value names its planned edit.
    """
    if not 2 <= clients <= 8 or not 1 <= rotations <= 4096:
        raise ValueError('long rotation profile requires 2..8 clients and 1..4096 rotations')
    seeds = [int.from_bytes(hashlib.sha256(f'long-rotation-v1:{master_seed}:{i}'.encode()).digest()[:8], 'big')
             for i in range(clients)]
    writers = []
    for writer, seed in enumerate(seeds):
        rng = random.Random(seed)
        writers.append([{'writer': writer, 'ordinal': step, 'edit': f'w{writer}:{step}',
                         'field': f'owned_{writer}_{step // 2 % 4}' if step % 2 == 0 else f'shared_{rng.randrange(3)}',
                         'value': (writer + 1) * 10000000 + step * 1024 + rng.randrange(1024)}
                        for step in range(rotations)])
    return {'schema_version': 1, 'scenario': 'public-plugin.long-rotations.v1', 'master_seed': master_seed,
            'writer_seeds': seeds, 'client_count': clients, 'writers': writers,
            'replay_scope': 'exact generated inputs; observed OS/gRPC commit order may differ'}


def ping_results(snapshot):
    """Finished Ping attempts by result, from one client's diagnostics."""
    return {item['result']: item['count'] for item in snapshot['metrics']['rpc_by_operation_result']
            if item['operation'] == 13}  # Public OPERATION_PING.


def soak_sample(pid, snapshot):
    """One soak writer's process figures and every runtime gauge (campaign.soak_gauges).

    anon is RssAnon: resident anonymous memory (heap, thread stacks and
    anonymous mappings). It excludes file-backed and shared pages, which the
    host may reclaim at any time, so reclaim cannot mask heap growth.
    On Windows anon is the private commit, rss and hwm the working set and its
    peak, and files the open handle count.

    On Linux, threads excludes gRPC's iomgr timer threads, which timer_threads
    counts by their thread name: gRPC starts another one whenever its last
    waiting timer thread runs expired timers, and keeps every one until gRPC
    shuts down, so their count is a step that depends on timing, not a leak.
    """
    if os.name == 'nt':
        return dict(**windows_process_figures(pid), gauges=campaign.soak_gauges(snapshot['metrics']))
    status = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
    threads, timer_threads = 0, 0
    for task in os.listdir(f'/proc/{pid}/task'):
        try:
            name = Path(f'/proc/{pid}/task/{task}/comm').read_text().strip()
        except OSError:
            continue  # The thread ended after the listing.
        timer_threads += name == campaign.GRPC_TIMER_THREAD
        threads += name != campaign.GRPC_TIMER_THREAD
    kib = {key: int(status[key].split()[0]) * 1024 for key in ('RssAnon', 'VmRSS', 'VmHWM')}
    return dict(anon=kib['RssAnon'], rss=kib['VmRSS'], hwm=kib['VmHWM'], threads=threads,
                timer_threads=timer_threads, files=len(os.listdir(f'/proc/{pid}/fd')),
                gauges=campaign.soak_gauges(snapshot['metrics']))


def windows_process_figures(pid):
    """Memory, thread and handle figures of process pid, through the Win32 API."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
                    ('PrivateUsage', ctypes.c_size_t)]

    class ThreadEntry(ctypes.Structure):
        _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD), ('th32ThreadID', wintypes.DWORD),
                    ('th32OwnerProcessID', wintypes.DWORD), ('tpBasePri', wintypes.LONG),
                    ('tpDeltaPri', wintypes.LONG), ('dwFlags', wintypes.DWORD)]

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    process = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION | VM_READ
    if not process:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        counters = Counters(cb=ctypes.sizeof(Counters))
        handles = wintypes.DWORD()
        if (not kernel32.K32GetProcessMemoryInfo(wintypes.HANDLE(process), ctypes.byref(counters), counters.cb)
                or not kernel32.GetProcessHandleCount(wintypes.HANDLE(process), ctypes.byref(handles))):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(wintypes.HANDLE(process))
    snapshot = kernel32.CreateToolhelp32Snapshot(0x4, 0)  # TH32CS_SNAPTHREAD
    if snapshot in (None, wintypes.HANDLE(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry, threads = ThreadEntry(dwSize=ctypes.sizeof(ThreadEntry)), 0
        more = kernel32.Thread32First(wintypes.HANDLE(snapshot), ctypes.byref(entry))
        while more:
            threads += entry.th32OwnerProcessID == pid
            more = kernel32.Thread32Next(wintypes.HANDLE(snapshot), ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(wintypes.HANDLE(snapshot))
    return dict(anon=counters.PrivateUsage, rss=counters.WorkingSetSize, hwm=counters.PeakWorkingSetSize,
                threads=threads, files=handles.value)


async def idle_executor_snapshot(driver):
    """A diagnostics snapshot taken once both executors have no queued or running job.

    A job can still be accounted as running just after its callback answered
    the driver; a job that never finishes fails the watchdog.
    """
    deadline = time.monotonic() + 10  # Failure-only watchdog.
    while True:
        snapshot = (await driver.command('diagnostics'))['snapshot']
        executors = snapshot['metrics']['executors']
        if all(executor['queued'] == 0 and executor['running'] == 0 for executor in executors.values()):
            return snapshot
        if time.monotonic() >= deadline:
            raise AssertionError(f'executors stayed busy: {executors}')


# Per-gauge change one unary Ping makes while it is in flight: one RPC attempt,
# and completion-queue tags where StartCall and Finish share the call's single
# preallocated tag (submit in source/live/runtime/LiveUnary.cpp).
PING_TAGS = 1
PING_GAUGES = {'rpc_in_flight': 1, 'cq_tags.outstanding': PING_TAGS, 'cq_tags.handling': PING_TAGS,
               'cq_tags.retained': PING_TAGS}
TIMING_SLOT = 'capacity.request_slots.timing.current'


def soak_drift(baseline, sample):
    """Gauges that differ from the warm-up baseline, as {name: [baseline, value]}.

    The timing slot holds 0 or 1 Ping; while its count differs from the
    baseline's, each PING_GAUGES gauge may differ by that Ping's share in the
    same direction and no further.
    """
    timing, before = sample['gauges'][TIMING_SLOT], baseline['gauges'][TIMING_SLOT]
    drift = {}
    for name, value in sample['gauges'].items():
        expected = baseline['gauges'].get(name)
        if name == TIMING_SLOT:
            held = value in (0, 1)
        elif name in PING_GAUGES and timing != before:
            held = 0 <= (value - expected if timing > before else expected - value) <= PING_GAUGES[name]
        else:
            held = value == expected
        if not held:
            drift[name] = [expected, value]
    return drift


def rss_slope(points):
    """Least-squares slope of (rotation, bytes) points, in bytes per rotation."""
    count = len(points)
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    spread = sum((x - mean_x) ** 2 for x, _ in points)
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / spread if spread else 0.0


class RotationRun:
    """Writers rotating between two backends behind one RotatingProxy.

    Each rotation parks new connections, so every writer reconnects once,
    queues one planned edit without issuing it, recovers on the other backend
    and converges before the rotation ends.
    """
    def __init__(self, test, plan, proxy, backends, history):
        self.test, self.plan, self.proxy, self.backends, self.history = test, plan, proxy, backends, history
        self.clients = plan['client_count']
        self.drivers, self.status, self.generations = [], [], []
        self.seen, self.initial_ms = [0] * self.clients, [0] * self.clients
        self.model, self.expected, self.identities = Model(), {}, {}

    def record(self, kind, **fields):
        self.history.write(json.dumps(dict(event=kind, **fields), sort_keys=True, separators=(',', ':')) + '\n')
        self.history.flush()

    async def read_records(self, index, driver):
        """This writer's read-stream records since its previous call, and its UPDATE attempts."""
        test = self.test
        snapshot = (await driver.command('diagnostics-history'))['snapshot']
        records = [r for r in snapshot['recent_records'] if r['sequence'] > self.seen[index]]
        # The bounded history still holds every record since the last call.
        test.assertTrue(not snapshot['recent_records'] or
                        snapshot['recent_records'][0]['sequence'] <= self.seen[index] + 1, snapshot.get('history'))
        if snapshot['recent_records']:
            self.seen[index] = snapshot['recent_records'][-1]['sequence']
        attempts = sum(item['count'] for item in snapshot['metrics']['rpc_attempts_by_operation']
                       if item['operation'] == 4)  # Public OPERATION_UPDATE.
        configuration = snapshot['effective_configuration']
        # A first reconnect waits less than twice reconnect_initial_ms; the
        # maximum admits a doubled step, whose window is twice as wide.
        test.assertGreaterEqual(configuration['reconnect_maximum_ms'], 2 * configuration['reconnect_initial_ms'],
                                configuration)
        self.initial_ms[index] = configuration['reconnect_initial_ms']
        return [r for r in records if r['subject_id'] == int(self.status[index]['subject']) and
                r['fields'].get('operation') == 3], attempts  # Public OPERATION_READ.

    async def join(self, drivers):
        """Every writer is READY with one read stream and no UPDATE attempt."""
        for index, driver in enumerate(drivers):
            self.drivers.append(driver)
            self.status.append(await driver.command('observe-status'))
            reads, attempts = await self.read_records(index, driver)
            resumed = [r['fields']['stream_generation'] for r in reads if r['event_name'] == 'stream_resumed']
            self.test.assertEqual((len(resumed), attempts), (1, 0), reads)
            self.generations.append(resumed[0])
            self.record('writer-ready', writer=index, pid=driver.process.pid, generation=resumed[0])

    async def rotate(self, rotation, *, gated, relaxed=False, client_updates=None):
        """One outage and recovery with exact accounting of the writers' planned edits.

        gated holds every caught-up read reply until all writers reconnected,
        with no write issued. relaxed admits the extra retries of a reconnect
        whose credentials the service rejected: at least one retry_scheduled
        and a changed incident. client_updates are the UPDATE attempts each
        writer has issued after this rotation, the rotation number by default.
        """
        test, state, clients, drivers = self.test, self.test.state, self.clients, self.drivers
        name = 'B' if rotation % 2 else 'A'
        before = len(test.commits())
        read_begin = len(state.scene_reads)
        read_attempt = state.attempts.get('SubscribeReadLayer', 0)
        write_attempt = state.attempts.get('UpdateLayer', 0)
        read_faults = [state.plan('SubscribeReadLayer', 'before_reply', attempt=read_attempt + i + 1,
                                  pause=True) for i in range(clients)] if gated else []
        try:
            retired = await asyncio.wait_for(self.proxy.rotate_and_wait(self.backends[name], pause=True), 5)
            test.assertGreaterEqual(retired, clients)
            for index, driver in enumerate(drivers):
                current = await driver.command('await-status 3')
                test.assertEqual((current['subject'], current['incident']),
                                 (self.status[index]['subject'], self.status[index]['incident']), current)
                test.assertGreater(current['revision'], self.status[index]['revision'])
                self.status[index] = current
            ops = [writer[rotation - 1] for writer in self.plan['writers']]
            for index, driver in enumerate(drivers):
                # During the outage the edit is admitted and queued, never issued.
                result = await driver.command(f"usd-set-admit 1 {ops[index]['field']} {ops[index]['value']}")
                test.assertEqual((result['result'], result['status_result'], result['pending']), (0, 8, 1), result)
            test.assertEqual(len(test.commits()), before)
            test.assertEqual(state.attempts.get('UpdateLayer', 0), write_attempt)
            self.proxy.gate.set()
            if gated:
                await asyncio.wait_for(asyncio.gather(*(fault.reached.wait() for fault in read_faults)), 5)
                # Every caught-up reply is held with no write issued.
                test.assertEqual(state.attempts.get('UpdateLayer', 0), write_attempt)
                for fault in read_faults:
                    fault.release.set()
            for index, driver in enumerate(drivers):
                current = await driver.command('await-status 2')
                test.assertEqual(current['subject'], self.status[index]['subject'], current)
                if not relaxed:
                    test.assertEqual(current['incident'], self.status[index]['incident'], current)
                self.status[index] = current
        finally:
            self.proxy.gate.set()
            for fault in read_faults:
                fault.release.set()
        reads = state.scene_reads[read_begin:]
        test.assertEqual(len(reads), clients)
        test.assertEqual(len({read['writer'] for read in reads}), clients)
        test.assertTrue(all(read['backend'] == name and read['cursor'] == before and read['object'] == 1
                            for read in reads), reads)
        saves = await writer_tasks(*(driver.command('save 5000') for driver in drivers))
        test.assertTrue(all(result['result'] == 0 for result in saves), saves)
        commits = test.commits()[before:]
        test.assertEqual(len(commits), clients)
        decoded = [{'sequence': commit.sequence, 'writer': commit.writer, 'fields': decode_update(commit.payload)}
                   for commit in commits]
        partial = dict(self.plan, writers=[[op] for op in ops])
        changed, _ = account_writers(partial, [{'edit': op['edit'], 'ordinal': 1} for op in ops], decoded,
                                    base_sequence=before)
        owners = {(op['field'], op['value']): op['writer'] for op in ops}
        for commit in decoded:
            writer = owners[next(iter(commit['fields'].items()))]
            test.assertEqual(self.identities.setdefault(writer, commit['writer']), commit['writer'],
                             'writer identity changed across rotations')
        for commit in commits:
            self.model.apply(commit.payload, commit.sequence)
        self.expected.update(changed)
        test.assertEqual(self.expected, self.model.state)
        retries, resumed_generations = [], []
        for index, driver in enumerate(drivers):
            current = await test.wait_fields(driver, self.expected)
            test.assertEqual((current['result'], current['pending'], current['subject']),
                             (0, 0, self.status[index]['subject']), current)
            reads, attempts = await self.read_records(index, driver)
            test.assertEqual(attempts, rotation if client_updates is None else client_updates[index],
                             'client UPDATE attempts differ from admissions')
            retry = [r['fields'] for r in reads if r['event_name'] == 'retry_scheduled']
            resumed = [r['fields'] for r in reads if r['event_name'] == 'stream_resumed']
            if relaxed:
                test.assertEqual(len(resumed), 1, reads)
                test.assertGreaterEqual(len(retry), 1, reads)
            else:
                test.assertEqual((len(retry), len(resumed)), (1, 1), reads)
                # One reconnect per rotation, drawn from the first backoff
                # window, below twice reconnect_initial_ms; an unreset count
                # would widen the window each rotation and fail the layer
                # after reconnect_max_attempts.
                test.assertEqual((retry[0]['attempt_id'], retry[0]['stream_generation']),
                                 (1, self.generations[index]), retry)
                test.assertLess(retry[0]['delay_ms'], 2 * self.initial_ms[index], retry)
            test.assertGreater(resumed[0]['stream_generation'], self.generations[index], resumed)
            test.assertEqual(resumed[0]['resume_sequence'], before, resumed)
            self.generations[index] = resumed[0]['stream_generation']
            retries.append([r['delay_ms'] for r in retry] if relaxed else retry[0]['delay_ms'])
            resumed_generations.append(self.generations[index])
        return dict(backend=name, gated=gated, retired=retired, boundary=before, generations=resumed_generations,
                    delay_ms=retries)

    def fields_digest(self):
        return hashlib.sha256(json.dumps(self.expected, sort_keys=True).encode()).hexdigest()[:16]


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.seed = getattr(self, 'seed_override', int(os.environ["LIVE_TEST_SEED"]))
        self.artifact = getattr(self, 'artifact_override', Path(os.environ["LIVE_TEST_ARTIFACT_DIR"]))
        self.state = (GraphState if getattr(self, 'graph_creation', False) else State)(self.seed)
        self.state.tokens = {"synthetic-writer": Principal("writer")}
        self.state.scene_reads = []
        self.state.partial_scene_reads = set()
        self.state.partial_scene_fired = []
        self.state.live_scene_hold = None
        self.drivers = []
        self.proxies, self.extra_servers = [], []
        self.trace = {"seed": self.seed, "states": []}
        self.server = (GraphServer(self.state, "A") if getattr(self, 'graph_creation', False) else
                       ReferenceServer(self.state, "A", service_factory=getattr(
                           self, 'writer_service_factory', scene_support.CompactSceneService)))
        self.addAsyncCleanup(self.cleanup)
        self.address = await self.server.start()
        self.channel = grpc.aio.insecure_channel(self.address)
        await asyncio.wait_for(self.channel.channel_ready(), 5)
        self.stub = rpc.LiveServiceStub(self.channel)
        self.session = (await self.stub.CreateSession(pb.CreateSessionRequest(path="plugin", name="public-consumer"),
                                                     metadata=WRITER, timeout=5)).session_id
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name="root.live")),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=empty_snapshot()))]), metadata=WRITER, timeout=5)

    async def cleanup(self):
        failures = []
        for driver in self.drivers:
            if driver.process.returncode is None:
                try:
                    driver.process.kill()
                except ProcessLookupError:
                    pass
        try:
            for operations in ((d.process.wait() for d in self.drivers), (d.drainer for d in self.drivers),
                               (proxy.close() for proxy in self.proxies),
                               (self.channel.close(),) if hasattr(self, 'channel') else (),
                               (server.stop() for server in self.extra_servers), (self.server.stop(),)):
                try:
                    results = await asyncio.wait_for(asyncio.gather(*operations, return_exceptions=True), 5)
                    failures.extend(result for result in results if isinstance(result, BaseException))
                except Exception as error:
                    failures.append(error)
            try:
                self.state.assert_faults_fired()
            except Exception as error:
                failures.append(error)
        finally:
            self.trace.update(ledger=self.state.snapshot(), attempts=self.state.attempts,
                              reads=self.state.scene_reads, cleanup_errors=[repr(e) for e in failures])
            (self.artifact / (self.id().split(".")[-1] + ".plugin.json")).write_text(
                json.dumps(self.trace, indent=2, sort_keys=True) + "\n")
        self.assertFalse(failures, failures)

    def commits(self):
        return self.state.sessions[self.session].layers["root.live"].commits

    def oracle(self):
        model = Model()
        for commit in self.commits():
            model.apply(commit.payload, commit.sequence)
        return model.state

    async def peer(self, fields):
        await self.stub.UpdateLayer(iter([
            pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name="root.live",
                                                             object_id=1, client_id="independent-peer")),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=update(fields)))]), metadata=WRITER, timeout=5)

    async def tls_pki(self, label='ca'):
        pki = await tls_fixture.case_pki(self, label)
        self.pki = self.pkis.get('ca')
        return pki

    async def start(self, address=None, environment=None):
        from tools.live.writer_campaign import admit_client
        admit_client(self)
        if getattr(self, 'transport_tls', False):  # The driver dials a TLS-terminating fixture proxy.
            environment = dict(environment or {}, **self.pki.environment())
        process = await asyncio.create_subprocess_exec(
            os.environ["LIVE_TEST_PLUGIN_DRIVER"], address or self.address, self.session, str(self.seed),
            os.environ["LIVE_TEST_PLUGIN_PLUG_INFO"], stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env=None if environment is None else dict(os.environ, **environment),
            limit=1024 * 1024)  # A diagnostics-history reply is one line of up to 256 KiB.
        driver = scene_support.Driver(process, self.artifact / f"{self.id().split('.')[-1]}-{len(self.drivers)}.plugin.log")
        self.drivers.append(driver)
        ready = await driver.receive()
        self.assertEqual(ready["type"], "ready", ready)
        self.assertEqual((ready["result"], ready["state"], ready["pending"]), (0, 2, 0))
        self.assertEqual(ready["fields"], self.oracle())
        self.trace["states"].append(ready)
        return driver

    async def wait_fields(self, driver, expected):
        command = "wait " + str(len(expected)) + "".join(f" {key} {value}" for key, value in sorted(expected.items()))
        result = await driver.command(command)
        self.assertEqual(result["fields"], expected)
        self.trace["states"].append(result)
        return result

    async def close(self, driver):
        result = await driver.command("close")
        await asyncio.wait_for(driver.process.wait(), 10)
        await driver.drainer
        self.assertIsNone(driver.failure)
        self.assertEqual(driver.process.returncode, 0, driver.log.read_text()[-8192:])
        self.trace["states"].append(result)
        return result

    async def test_direct_usd_authoring_batches_change_blocks_and_converges(self):
        await self.peer({"initial": 7})
        first, second = await self.start(), await self.start()
        expected = {"initial": 7}
        updates_before = self.state.attempts["UpdateLayer"]
        # Review focus 3: an empty change block issues nothing.
        self.assertEqual((await first.command("usd-set 0"))["result"], 0)
        self.assertEqual((await first.command("save 5000"))["result"], 0)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates_before)
        # One change block with two fields -> exactly one UpdateLayer.
        self.assertEqual((await first.command("usd-set 2 a 1 b 2"))["result"], 0)
        self.assertEqual((await first.command("save 5000"))["result"], 0)
        expected.update({"a": 1, "b": 2})
        self.assertEqual(self.state.attempts["UpdateLayer"], updates_before + 1)
        self.assertEqual(decode_update(self.commits()[-1].payload), {"a": 1, "b": 2})
        for driver in (first, second):
            await self.wait_fields(driver, expected)
        # Review focus 2: two change blocks before any wait -> two ordered updates.
        self.assertEqual((await second.command("usd-set 1 c 3"))["result"], 0)
        self.assertEqual((await second.command("usd-set 1 d 4"))["result"], 0)
        self.assertEqual((await second.command("save 5000"))["result"], 0)
        expected.update({"c": 3, "d": 4})
        self.assertEqual(self.state.attempts["UpdateLayer"], updates_before + 3)
        self.assertEqual([decode_update(c.payload) for c in self.commits()[-2:]], [{"c": 3}, {"d": 4}])
        for driver in (first, second):
            await self.wait_fields(driver, expected)
        # Review focus 5: same-field concurrent edit converges to service order.
        self.assertEqual((await first.command("usd-set 1 shared 100"))["result"], 0)
        self.assertEqual((await second.command("usd-set 1 shared 200"))["result"], 0)
        for driver in (first, second):
            self.assertEqual((await driver.command("save 5000"))["result"], 0)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates_before + 5)
        self.assertEqual(sorted(decode_update(c.payload)["shared"] for c in self.commits()[-2:]), [100, 200])
        expected["shared"] = self.oracle()["shared"]
        self.assertIn(expected["shared"], (100, 200))
        for driver in (first, second):
            await self.wait_fields(driver, expected)
        fresh = await self.start()
        await self.wait_fields(fresh, expected)
        self.assertEqual(self.oracle(), expected)
        for driver in (first, second, fresh):
            self.assertEqual((await self.close(driver))["fields"], expected)

    async def test_direct_usd_save_admits_host_edits_from_the_same_host_call(self):
        await self.peer({"initial": 7})
        driver = await self.start()
        expected = {"initial": 7}
        # Each command authors and saves inside one host call, before the flush
        # job its edit enqueued can run: only the save path admits the edit.
        for command, key in (("usd-set-save", "saved"), ("usd-set-sdf-save", "sdf_saved")):
            updates = self.state.attempts["UpdateLayer"]
            saved = await driver.command(f"{command} {key} 5")
            self.assertEqual(saved["offhost"], 14, saved)  # BUSY: only the host can admit the edit.
            self.assertEqual(saved["pending"], 1, saved)   # Admitted inside the save call.
            self.assertEqual((await driver.command("save 5000"))["result"], 0)
            self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)
            self.assertEqual(decode_update(self.commits()[-1].payload), {key: 5})
            expected[key] = 5
            await self.wait_fields(driver, expected)
        fresh = await self.start()
        await self.wait_fields(fresh, expected)
        self.assertEqual(self.oracle(), expected)
        for observer in (driver, fresh):
            self.assertEqual((await self.close(observer))["fields"], expected)

    async def test_cached_open_of_an_attached_layer_issues_no_stat(self):
        driver = await self.start()
        stats = self.state.attempts["StatLayer"]
        # A held StatLayer would stall any resolution that asks the service.
        # The attachment is the existence evidence, so the cached open by URI
        # and by alias completes without reaching the held reply.
        fault = self.state.plan("StatLayer", "before_reply", pause=True)
        self.addCleanup(fault.release.set)
        opened = await driver.command("cached-open")
        self.assertEqual((opened["type"], opened["result"], opened["state"]), ("cached-open", 0, 2), opened)
        self.assertFalse(fault.reached.is_set())
        self.assertEqual(self.state.attempts["StatLayer"], stats)
        self.state.faults.remove(fault)  # Deliberately unfired.
        await self.close(driver)

    async def test_sdf_save_of_an_attached_layer_reads_the_cached_timestamp(self):
        driver = await self.start()
        opened_at = self.commits()[-1].timestamp_us  # The modification time the first open's stat returned.
        stats = self.state.attempts["StatLayer"]
        saved = await driver.command("sdf-save-timestamp saved 5")
        self.assertEqual(saved["modified_us"], opened_at, saved)
        self.assertEqual(self.state.attempts["StatLayer"], stats)
        await self.wait_fields(driver, {"saved": 5})
        self.assertEqual((await self.close(driver))["fields"], {"saved": 5})

    async def test_reopen_of_a_released_layer_stats_again(self):
        driver = await self.start()
        ready = self.trace["states"][-1]
        stats = self.state.attempts["StatLayer"]
        # The released backing is closing or gone, never attached, so the
        # reopen's resolution asks the service again.
        reopened = await driver.command("reopen-root")
        self.assertEqual((reopened["type"], reopened["result"], reopened["state"]), ("reopened", 0, 2), reopened)
        self.assertNotEqual(reopened["subject"], ready["subject"])
        self.assertEqual(self.state.attempts["StatLayer"], stats + 1)
        await self.close(driver)

    async def test_direct_usd_many_fields_in_one_change_block_are_one_update(self):
        await self.peer({"initial": 7})
        writer = await self.start()
        updates, base = self.state.attempts["UpdateLayer"], len(self.commits())
        authored = await writer.command("usd-set-many 256")
        self.assertEqual((authored["result"], authored["status_result"], authored["count"]), (0, 0, 256), authored)
        self.assertEqual((await writer.command("save 5000"))["result"], 0)
        many = {f"many{i}": i + 1 for i in range(256)}
        self.assertEqual((self.state.attempts["UpdateLayer"], len(self.commits())), (updates + 1, base + 1))
        self.assertEqual(decode_update(self.commits()[-1].payload), many)
        expected = {"initial": 7, **many}
        await self.wait_fields(writer, expected)
        fresh = await self.start()
        await self.wait_fields(fresh, expected)
        self.assertEqual(self.oracle(), expected)
        for driver in (writer, fresh):
            self.assertEqual((await self.close(driver))["fields"], expected)

    async def test_direct_usd_edit_after_threaded_stage_composition_is_admitted(self):
        # A UsdStage composed over the live layer with at least two USD work
        # threads reads the layer's backing from worker threads. Those reads
        # overlap and retire in any order; none leaves nesting state behind,
        # so the next host edit is admitted rather than refused BUSY. The
        # driver requests two work threads when the limit is below two; USD
        # caps that request at the process's physical concurrency, and
        # PXR_WORK_THREAD_LIMIT overrides it.
        await self.peer({"initial": 7})
        writer = await self.start()
        updates, base = self.state.attempts["UpdateLayer"], len(self.commits())
        authored = await writer.command("usd-stage-set 256 composed 11")
        self.assertGreater(
            authored["concurrency"], 1,
            "threaded composition needs a USD work-thread limit >= 2; "
            f"got {authored['concurrency']}; unset PXR_WORK_THREAD_LIMIT or run on >= 2 cores: {authored}")
        self.assertEqual(authored["composed"], 256 * 5, authored)
        self.assertEqual((authored["result"], authored["status_result"], authored["state"]), (0, 0, 2), authored)
        self.assertEqual((await writer.command("save 5000"))["result"], 0)
        self.assertEqual((self.state.attempts["UpdateLayer"], len(self.commits())), (updates + 1, base + 1))
        self.assertEqual(decode_update(self.commits()[-1].payload), {"composed": 11})
        expected = {"initial": 7, "composed": 11}
        await self.wait_fields(writer, expected)
        fresh = await self.start()
        await self.wait_fields(fresh, expected)
        self.assertEqual(self.oracle(), expected)
        for driver in (writer, fresh):
            self.assertEqual((await self.close(driver))["fields"], expected)

    async def test_direct_usd_observer_receives_updates_and_never_writes(self):
        await self.peer({"initial": 7})
        reads = len(self.state.scene_reads)
        writer = await self.start()
        writer_ids = {read["writer"] for read in self.state.scene_reads[reads:]}
        reads = len(self.state.scene_reads)
        observer = await self.start()  # Mode 3: opens the live layer and never authors it.
        observer_ids = {read["writer"] for read in self.state.scene_reads[reads:]}
        self.assertEqual((len(writer_ids), len(observer_ids)), (1, 1), self.state.scene_reads)
        self.assertTrue(writer_ids.isdisjoint(observer_ids))
        base, updates = len(self.commits()), self.state.attempts["UpdateLayer"]
        expected = {"initial": 7}
        for fields in ({"a": 1}, {"b": 2, "c": 3}, {"a": 4}):
            command = f"usd-set {len(fields)}" + "".join(f" {key} {value}" for key, value in fields.items())
            self.assertEqual((await writer.command(command))["result"], 0)
            self.assertEqual((await writer.command("save 5000"))["result"], 0)
            expected.update(fields)
            await self.wait_fields(observer, expected)
        await self.peer({"peer": 9})
        expected["peer"] = 9
        for driver in (writer, observer):
            await self.wait_fields(driver, expected)
        # A save on the observer drains nothing and issues nothing.
        self.assertEqual((await observer.command("save 5000"))["result"], 0)
        # The ledger attributes every non-peer write to the writer process.
        self.assertEqual({commit.writer for commit in self.commits()[base:]}, writer_ids | {"independent-peer"})
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 4)  # Three change blocks and the peer.
        self.assertEqual((await client_update_attempts(writer), await client_update_attempts(observer)), (3, 0))
        state = await observer.command("state")
        self.assertEqual((state["result"], state["state"], state["pending"]), (0, 2, 0), state)
        self.assertEqual(self.oracle(), expected)
        for driver in (writer, observer):
            self.assertEqual((await self.close(driver))["fields"], expected)
        self.assertEqual(len(self.commits()), base + 4, "closing the observer issued a write")
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 4)

    async def test_packaged_plugin_public_facade_alias_and_fresh_observer(self):
        await self.peer({"initial": 7})
        first, second = await self.start(), await self.start()
        initial = {"initial": 7}
        attempts = dict(self.state.attempts)
        checked = await first.command("facade-checks")
        self.assertEqual(checked, {"type": "facade-checked", "fields": initial})
        self.assertEqual(self.state.attempts, attempts, "rejected/abandoned facade operations issued service work")
        self.assertEqual(self.oracle(), initial)
        expected, changes = dict(initial), [{"initial": 7}]
        generator = random.Random(self.seed)
        for index in range(4):
            driver = (first, second)[index % 2]
            if index == 2:
                await first.command("reset-context")  # Existing binding must retain its original host context.
            key, value = f"client_{index % 2}", generator.randint(-10000, 10000)
            updates = self.state.attempts["UpdateLayer"]
            authored = await driver.command(f"usd-set 1 {key} {value}")
            self.assertEqual((authored["result"], authored["status_result"]), (0, 0), authored)
            self.assertEqual((await driver.command("save 5000"))["result"], 0)
            # The saved change block is one admitted, issued mutation.
            self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)
            expected[key] = value
            changes.append({key: value})
            for observer in (first, second):
                await self.wait_fields(observer, expected)
            self.assertEqual(self.oracle(), expected)
        await self.peer({"after_reset": 91})
        expected["after_reset"] = 91
        changes.append({"after_reset": 91})
        for driver in (first, second):
            await self.wait_fields(driver, expected)
        fresh = await self.start()
        await self.wait_fields(fresh, expected)
        self.assertEqual([decode_update(c.payload) for c in self.commits()[1:]], changes)
        self.assertEqual((self.state.attempts["CreateLayer"], self.state.attempts["SubscribeReadLayer"],
                          self.state.attempts["UpdateLayer"]), (1, 3, len(changes)))
        self.assertEqual(len(self.state.scene_reads), 3)  # Alias lookup never creates another subscription.
        for driver in (first, second, fresh):
            self.assertEqual((await self.close(driver))["fields"], expected)

    async def test_seeded_public_writers_account_every_edit_and_conflict(self):
        replay = os.environ.get('LIVE_TEST_WRITER_PLAN')
        plan = load_writer_plan(Path(replay)) if replay else self.writer_inputs(writer_plan(
            self.seed, clients=getattr(self, 'writer_clients', 8), steps=getattr(self, 'writer_steps', 16)))
        # Explicit replay uses a fresh fixture too, so a named test-only service
        # defect can be reproduced without changing the shared default service.
        if replay:
            result = await self.writer_trial(plan, self.artifact / 'writer-replay')
            publish_writer(self.artifact / 'writer-replay-result.json', result)
            if result['outcome'] == 'mismatch':
                raise WriterMismatch(**result['signature'])
            self.assertEqual(result['outcome'], 'passed', result)
            return
        await self.run_writers_with_reduction(plan)

    async def run_writers_with_reduction(self, plan):
        try:
            await self.exercise_writers(plan)
        except WriterMismatch as error:
            # Retire the original fixture before launching any reduction trial.
            await self.cleanup()
            self._cleanups.clear()
            summary = await reduce_writers(plan, error.signature,
                                           self.artifact / 'writer-reduction', self.writer_trial)
            print('Writer reduction: ' + json.dumps(summary, sort_keys=True), flush=True)
            raise  # Reduction never turns the original failure into a pass.

    def writer_inputs(self, plan):
        plan = copy.deepcopy(plan)
        plan['initial_fixture_sha256'] = hashlib.sha256(empty_snapshot()).hexdigest()
        candidate = Path(os.environ['LIVE_TEST_PLUGIN_DRIVER'])
        plan['candidate'] = json.loads(Path(str(candidate) + '.manifest.json').read_text())
        plan['source_fingerprint'] = os.environ['LIVE_TEST_SOURCE_FINGERPRINT']
        plan['reference_identity'] = json.loads(os.environ['LIVE_TEST_REFERENCE_IDENTITY'])
        from tools.live.sanitizer import recorded
        plan['sanitizer_runtime'] = recorded(os.environ)
        return validate_writer_plan(plan)

    async def writer_trial(self, plan, directory, *, automatic_reduction=False):
        validate_writer_plan(plan)
        directory.mkdir(parents=True, exist_ok=True)
        fixture = PluginTests('test_seeded_public_writers_account_every_edit_and_conflict')
        fixture.artifact_override = directory
        fixture.seed_override = plan['master_seed']
        if plan.get('test_service_fault') == 'corrupt-owned-0-2':
            fixture.writer_service_factory = CorruptWriterService
        result = dict(outcome='unresolved', cleanup_complete=False)
        try:
            await fixture.asyncSetUp()
            if automatic_reduction:
                await fixture.run_writers_with_reduction(plan)
            else:
                await fixture.exercise_writers(plan)
            result.update(outcome='passed')
        except WriterMismatch as error:
            result.update(outcome='mismatch', signature=error.signature)
        except Exception as error:
            result.update(error=repr(error))
        finally:
            try:
                # Run all registered cleanups, even on partial setup failure.
                while fixture._cleanups:
                    callback, args, kwargs = fixture._cleanups.pop()
                    await callback(*args, **kwargs)
                result['cleanup_complete'] = True
            except Exception as error:
                result.update(outcome='unresolved', cleanup_complete=False, cleanup_error=repr(error))
        return result

    async def exercise_writers(self, plan):
        # The campaign binds one intent per command and holds each writer's first RPC: one unchained write at a time.
        self.state.advertise_capabilities = False
        validate_writer_plan(plan)
        from tools.live.writer_campaign import observe_workload
        observe_workload(self, plan)
        prefix = self.artifact / self.id().split('.')[-1]
        Path(str(prefix) + '.plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
        # Both header and full inputs exist before any native writer starts.
        print(f"many-writer master_seed={self.seed} clients={plan['client_count']} plan={prefix}.plan.json", flush=True)
        self.trace['writer_plan'] = plan
        admissions = []
        faults = []
        with Path(str(prefix) + '.history.jsonl').open('w') as history:
            def record(kind, **fields):
                history.write(json.dumps(dict(event=kind, **fields), sort_keys=True) + '\n')
                history.flush()  # Partial failure retains the prefix that actually ran.

            record('plan', master_seed=self.seed, scenario=plan['scenario'], writer_seeds=plan['writer_seeds'])
            drivers = [await self.start() for _ in plan['writers']]
            for index, driver in enumerate(drivers):
                record('writer-ready', writer=index, pid=driver.process.pid)
            faults = [self.state.plan('UpdateLayer', 'before', attempt=i + 1, pause=True)
                      for i in range(len(drivers))]

            admitted = [0] * len(drivers)

            async def author(index, driver):
                for op in plan['writers'][index]:
                    record('submit', **op)
                    # One plain USD change block. Its flush job has run when the
                    # command returns, so the block is one admitted intent; with
                    # every first RPC held, no admitted intent retires.
                    result = await driver.command(f"usd-set-admit 1 {op['field']} {op['value']}")
                    self.assertEqual((result['result'], result['status_result']), (0, 0), result)
                    admitted[index] += 1
                    self.assertEqual(result['pending'], admitted[index], result)
                    admission = {'edit': op['edit'], 'ordinal': admitted[index]}
                    admissions.append(admission)
                    record('admitted', writer=index, **admission)

            try:
                async with asyncio.TaskGroup() as authors:
                    for i, driver in enumerate(drivers):
                        authors.create_task(author(i, driver))
                await asyncio.wait_for(asyncio.gather(*(fault.reached.wait() for fault in faults)), 5)
                record('all-first-rpcs-held', attempts=self.state.attempts['UpdateLayer'])
                self.assertEqual(self.state.attempts['UpdateLayer'], len(drivers))
                self.assertEqual(len(self.commits()), 1, 'held requests committed before release')
                # All independent processes have admitted a complete queue while
                # every writer's first issued RPC is simultaneously held.
                for i, driver in enumerate(drivers):
                    state = await driver.command('state')
                    self.assertEqual(state['pending'], len(plan['writers'][i]), state)
                    own = {op['field']: op['value'] for op in plan['writers'][i]}
                    # The host content holds every edit this writer authored.
                    self.assertEqual(state['fields'], own)
                    # Each writer's first update is issued and held; the rest wait.
                    self.assertEqual(await client_update_attempts(driver), 1)
                    record('held-state', writer=i, state=state)
                late = await self.start()
                record('late-observer-ready', pid=late.process.pid)
            finally:
                for index in plan['release_order']:
                    record('release-first-rpc', attempt=index + 1)
                    faults[index].release.set()

            boundary = 1 + sum(map(len, plan['writers']))
            # Observe service progress explicitly before checking local save
            # retirement. A correctness campaign must not require hundreds of
            # instrumented, contending writes to finish in a save's five-second
            # deadline. The condition is notified by actual commits; timeout is
            # only a failure watchdog, never evidence of successful progress.
            async with asyncio.timeout(30):
                async with self.state.changed:
                    await self.state.changed.wait_for(lambda: len(self.commits()) >= boundary)
            record('service-commit-boundary', sequence=boundary, observed=len(self.commits()))
            # The larger nightly population includes >16,000 remote USD
            # applications under sanitizers. Preserve one event-driven save per
            # client with a workload-sized failure watchdog, not a speed gate.
            save_timeout_ms = 30000 if len(drivers) > 8 else 5000
            saves = await writer_tasks(*(driver.command(f'save {save_timeout_ms}',
                                                        timeout=save_timeout_ms / 1000 + 5)
                                         for driver in drivers))
            for index, result in enumerate(saves):
                record('save-completed', writer=index, result=result)
            if any(result['result'] != 0 for result in saves):
                async def capture(index, driver):
                    record('save-failure-state', writer=index, state=await driver.command('state'))
                    record('save-failure-diagnostics', writer=index, diagnostics=await driver.command('diagnostics-history'))
                await writer_tasks(*(capture(i, driver) for i, driver in enumerate(drivers)))
            self.assertTrue(all(result['result'] == 0 for result in saves), saves)
            commits = [{'sequence': commit.sequence, 'writer': commit.writer, 'fields': decode_update(commit.payload)}
                       for commit in self.commits()[1:]]
            record('commit-boundary', sequence=boundary, commits=commits)
            try:
                expected, lineage = account_writers(plan, admissions, commits)
            except WriterMismatch:
                # A semantic failure is only reproducible if every native
                # process can close cleanly; a crash cannot masquerade as loss.
                await writer_tasks(*(self.close(driver) for driver in drivers + [late]))
                raise
            self.assertEqual(len(self.commits()), boundary)
            self.assertEqual(self.state.attempts['UpdateLayer'], boundary - 1)
            self.assertEqual(expected, self.oracle())
            record('edit-accounting', lineage=lineage, expected=expected)
            for i, driver in enumerate(drivers + [late]):
                state = await self.wait_fields(driver, expected)
                self.assertEqual((state['result'], state['pending']), (0, 0), state)
                record('converged', writer=i, state=state)
            # Each change block issued exactly one UpdateLayer; the late observer none.
            for i, driver in enumerate(drivers):
                self.assertEqual(await client_update_attempts(driver), len(plan['writers'][i]))
            self.assertEqual(await client_update_attempts(late), 0)
            fresh = await self.start()
            state = await self.wait_fields(fresh, expected)
            self.assertEqual(state['pending'], 0, state)
            record('fresh-observer-converged', state=state)
            self.assertEqual(await client_update_attempts(fresh), 0)
            closed = await writer_tasks(*(self.close(driver) for driver in drivers + [late, fresh]))
            self.assertTrue(all(result['fields'] == expected for result in closed))
            self.assertEqual(len(self.commits()), boundary, 'teardown issued another mutation')
            record('complete', admitted=len(admissions), boundary=boundary)

    async def test_public_writer_reduction_reexecutes_real_clients_and_service(self):
        plan = self.writer_inputs(writer_plan(self.seed, clients=2, steps=4))
        plan['test_service_fault'] = 'corrupt-owned-0-2'
        directory = self.artifact / 'writer-reducer-selftest'
        first = await self.writer_trial(plan, directory, automatic_reduction=True)
        self.assertEqual(first['outcome'], 'mismatch', first)
        self.assertTrue(first['cleanup_complete'])
        summary = json.loads((directory / 'writer-reduction/reduction.json').read_text())
        self.assertEqual(summary['status'], 'complete', summary)
        self.assertTrue(summary['verified'] and summary['operation_deletion_minimal'])
        best = load_writer_plan(directory / 'writer-reduction/best.plan.json')
        self.assertEqual(sum(map(len, best['writers'])), 2, best)
        self.assertEqual(best['writers'][0][0]['edit'], 'w0:2')
        self.assertEqual(plan['writers'], load_writer_plan(directory / 'writer-reduction/original.plan.json')['writers'])
        replay = await self.writer_trial(best, directory / 'replay')
        self.assertEqual(replay, first)
        repaired = copy.deepcopy(best)
        del repaired['test_service_fault']
        passed = await self.writer_trial(repaired, directory / 'correct-service')
        self.assertEqual(passed, dict(outcome='passed', cleanup_complete=True))

    async def test_seeded_writers_resume_queued_edits_through_two_tls_backend_rotations(self):
        self.transport_tls = True
        await self.test_seeded_writers_resume_queued_edits_through_two_backend_rotations()

    async def test_seeded_writers_resume_queued_edits_through_two_backend_rotations(self):
        plan = writer_plan(self.seed, clients=4, steps=12)
        from tools.live.writer_campaign import observe_workload
        observe_workload(self, plan)
        plan.update(scenario='public-plugin.queued-rotations.v1',
                    phases=[{'backend': name, 'first_ordinal': phase * 4, 'end_ordinal': (phase + 1) * 4}
                            for phase, name in enumerate(('A', 'B', 'A'))],
                    initial_fixture_sha256=hashlib.sha256(empty_snapshot()).hexdigest(),
                    candidate=json.loads(Path(os.environ['LIVE_TEST_PLUGIN_DRIVER'] + '.manifest.json').read_text()))
        prefix = self.artifact / self.id().split('.')[-1]
        Path(str(prefix) + '.plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
        print(f"rotation-writers master_seed={self.seed} clients=4 plan={prefix}.plan.json", flush=True)
        self.trace['writer_plan'] = plan
        backend = ReferenceServer(self.state, 'B', service_factory=scene_support.CompactSceneService)
        self.extra_servers.append(backend)
        other = await backend.start()
        proxy = scene_support.RotatingProxy(self.address, ssl_context=(await self.tls_pki()).server_context()
                                            if getattr(self, 'transport_tls', False) else None)
        self.proxies.append(proxy)
        address = await proxy.start()
        admissions = []
        admitted = [0] * len(plan['writers'])
        with Path(str(prefix) + '.history.jsonl').open('w') as history:
            def record(kind, **fields):
                history.write(json.dumps(dict(event=kind, **fields), sort_keys=True) + '\n')
                history.flush()

            async def check_attempts(driver, index, phase, boundary, expected, issuing=False):
                # While a writer is `issuing` its window, its worker issues each
                # queued write after its local publication, so the count is
                # polled until it reaches the window under a failure-only watchdog.
                deadline = asyncio.get_running_loop().time() + 5
                while True:
                    snapshot = (await driver.command('diagnostics'))['snapshot']
                    attempts = sum(item['count'] for item in snapshot['metrics']['rpc_attempts_by_operation']
                                   if item['operation'] == 4)  # Public OPERATION_UPDATE.
                    if not issuing or attempts >= expected or asyncio.get_running_loop().time() > deadline:
                        break
                record('client-update-attempts', writer=index, phase=phase, boundary=boundary, count=attempts)
                self.assertEqual(attempts, expected, (index, phase, boundary, snapshot))

            record('plan', master_seed=self.seed, scenario=plan['scenario'], writer_seeds=plan['writer_seeds'])
            drivers = [await self.start(address) for _ in plan['writers']]
            identities = []
            for index, driver in enumerate(drivers):
                identities.append(await driver.command('observe-status'))
                record('writer-ready', writer=index, pid=driver.process.pid, status=identities[-1])

            for phase, definition in enumerate(plan['phases']):
                first, end = definition['first_ordinal'], definition['end_ordinal']
                before = len(self.commits())
                read_begin = len(self.state.scene_reads)
                read_faults, write_faults = [], []
                try:
                    if phase:
                        read_attempt = self.state.attempts['SubscribeReadLayer']
                        write_attempt = self.state.attempts['UpdateLayer']
                        read_faults = [self.state.plan('SubscribeReadLayer', 'before_reply',
                                                      attempt=read_attempt + i + 1, pause=True) for i in range(4)]
                        # Every queued write of the phase issues within its
                        # writer's window, so each attempt is held before the
                        # service evaluates it.
                        write_faults = [self.state.plan('UpdateLayer', 'before', attempt=write_attempt + i + 1,
                                                        pause=True) for i in range(4 * (end - first))]
                        retired = await asyncio.wait_for(proxy.rotate_and_wait(
                            other if phase == 1 else self.address, pause=True), 5)
                        self.assertGreaterEqual(retired, 4)
                        record('rotate', phase=phase, backend=definition['backend'], boundary=before,
                               retired_forwards=retired)
                        for index, driver in enumerate(drivers):
                            state = await driver.command('await-status 3')
                            self.assertEqual(state['subject'], identities[index]['subject'])
                            self.assertEqual(state['incident'], identities[index]['incident'])
                            self.assertGreater(state['revision'], identities[index]['revision'])
                            record('reconnecting', phase=phase, writer=index, status=state)

                    async def author(index, driver):
                        for op in plan['writers'][index][first:end]:
                            record('submit', phase=phase, **op)
                            # One plain USD change block; its flush job has run
                            # when the command returns.
                            result = await driver.command(f"usd-set-admit 1 {op['field']} {op['value']}")
                            admitted[index] += 1
                            # Every earlier phase's intents retired at its save.
                            # During an outage the status reports UNAVAILABLE and
                            # nothing retires, so each block adds exactly one
                            # pending intent; the first phase's connected updates
                            # may retire at any point.
                            if phase:
                                self.assertEqual((result['result'], result['status_result'], result['pending']),
                                                 (0, 8, admitted[index] - first), result)
                            else:
                                self.assertEqual((result['result'], result['status_result']), (0, 0), result)
                                self.assertLessEqual(result['pending'], admitted[index] - first, result)
                            admission = {'edit': op['edit'], 'ordinal': admitted[index]}
                            admissions.append(admission)
                            record('admitted', phase=phase, writer=index, **admission)

                    async with asyncio.TaskGroup() as authors:
                        for index, driver in enumerate(drivers):
                            authors.create_task(author(index, driver))
                    if phase:
                        expected = self.oracle()
                        for index, driver in enumerate(drivers):
                            state = await driver.command('state')
                            own = dict(expected)
                            own.update({op['field']: op['value'] for op in plan['writers'][index][first:end]})
                            self.assertEqual(state['fields'], own)
                            self.assertEqual(state['pending'], end - first)
                            self.assertEqual(state['subject'], identities[index]['subject'])
                            record('queued-during-outage', phase=phase, writer=index, state=state)
                            await check_attempts(driver, index, phase, 'proxy-paused', first)
                        self.assertEqual(len(self.commits()), before)
                        self.assertEqual(self.state.attempts['UpdateLayer'], write_attempt)
                        proxy.gate.set()
                        await asyncio.wait_for(asyncio.gather(*(fault.reached.wait() for fault in read_faults)), 5)
                        reads = self.state.scene_reads[read_begin:]
                        self.assertEqual(len(reads), 4)
                        self.assertEqual(len({read['writer'] for read in reads}), 4)
                        self.assertTrue(all(read['backend'] == definition['backend'] and
                                            read['cursor'] == before and read['object'] == 1 for read in reads), reads)
                        record('all-caught-up-replies-held', phase=phase, reads=reads, boundary=before)
                        self.assertEqual(self.state.attempts['UpdateLayer'], write_attempt)
                        for index, driver in enumerate(drivers):
                            await check_attempts(driver, index, phase, 'replies-held', first)
                        # Each first response was selected at the unchanged commit
                        # boundary. First-write gates keep it unchanged until all
                        # clients have consumed their zero-byte recovery reply.
                        for fault in read_faults:
                            fault.release.set()
                        await asyncio.wait_for(asyncio.gather(*(fault.reached.wait() for fault in write_faults)), 5)
                        self.assertEqual(len(self.commits()), before)
                        for index, driver in enumerate(drivers):
                            state = await driver.command('await-status 2')
                            self.assertEqual((state['subject'], state['incident']),
                                             (identities[index]['subject'], identities[index]['incident']))
                            identities[index] = state
                            record('recovered', phase=phase, writer=index, status=state)
                            # The endpoint chains writes, so the queued writes
                            # issue together within the window; the service
                            # holds each writer's first and its successors wait.
                            await check_attempts(driver, index, phase, 'window-issued', end, issuing=True)
                finally:
                    proxy.gate.set()
                    for fault in read_faults:
                        fault.release.set()
                    for index in plan['release_order']:
                        for fault in write_faults[index::4]:
                            record('release-write', phase=phase, attempt=fault.attempt)
                            fault.release.set()

                saves = await asyncio.gather(*(driver.command('save 5000') for driver in drivers))
                self.assertTrue(all(result['result'] == 0 for result in saves), saves)
                self.assertEqual(len(self.commits()), 1 + end * 4)
                commits = [{'sequence': commit.sequence, 'writer': commit.writer, 'fields': decode_update(commit.payload)}
                           for commit in self.commits()[1:]]
                partial = dict(plan, writers=[writer[:end] for writer in plan['writers']])
                expected, lineage = account_writers(partial, admissions, commits)
                self.assertEqual(expected, self.oracle())
                record('commit-boundary', phase=phase, commits=commits, lineage=lineage, expected=expected)
                for index, driver in enumerate(drivers):
                    state = await self.wait_fields(driver, expected)
                    self.assertEqual((state['result'], state['pending'], state['subject']),
                                     (0, 0, identities[index]['subject']))
                    record('converged', phase=phase, writer=index, state=state)
                    await check_attempts(driver, index, phase, 'converged', end)
                self.assertEqual(self.state.attempts['UpdateLayer'], end * 4)

            fresh = await self.start(address)
            await self.wait_fields(fresh, expected)
            self.assertEqual(self.state.attempts['CreateLayer'], 1)
            self.assertEqual(self.state.attempts['SubscribeReadLayer'], 13)
            closed = await asyncio.gather(*(self.close(driver) for driver in drivers + [fresh]))
            self.assertTrue(all(result['fields'] == expected for result in closed))
            self.assertEqual(len(self.commits()), 49)
            record('complete', admitted=len(admissions), boundary=49, rotations=2)

    async def rotation_setup(self, plan, label):
        """Write the plan artifact, then start backend B and the RotatingProxy in front of A."""
        campaign.observe_workload(self, plan)
        plan.update(initial_fixture_sha256=hashlib.sha256(empty_snapshot()).hexdigest(),
                    candidate=json.loads(Path(os.environ['LIVE_TEST_PLUGIN_DRIVER'] + '.manifest.json').read_text()))
        prefix = self.artifact / self.id().split('.')[-1]
        Path(str(prefix) + '.plan.json').write_text(json.dumps(plan, sort_keys=True) + '\n')
        print(f"{label} master_seed={self.seed} clients={plan['client_count']} rotations={len(plan['writers'][0])} "
              f"plan={prefix}.plan.json", flush=True)
        backend = ReferenceServer(self.state, 'B', service_factory=scene_support.CompactSceneService)
        self.extra_servers.append(backend)
        backends = {'A': self.address, 'B': await backend.start()}
        proxy = scene_support.RotatingProxy(self.address)
        self.proxies.append(proxy)
        return prefix, proxy, backends, await proxy.start()

    async def test_seeded_writers_converge_through_many_backend_rotations(self):
        """Separated rotations never accumulate toward the reconnect attempt limit.

        Each rotation parks new connections, so every writer reconnects once,
        queues one edit without issuing it, recovers on the other backend and
        converges before the next rotation. Each rotation's read retry is
        scheduled at the first backoff step, which shows the reconnect attempt
        count resets at every recovery readiness. The caught-up reply gate is
        armed only every 50th rotation and on the last, because the service
        scans every planned fault on each RPC.
        """
        clients = getattr(self, 'writer_clients', 4)
        rotations = getattr(self, 'writer_steps', 8)
        plan = rotation_writer_plan(self.seed, clients, rotations)
        prefix, proxy, backends, address = await self.rotation_setup(plan, 'long-rotation-writers')
        with Path(str(prefix) + '.history.jsonl').open('w') as history:
            run = RotationRun(self, plan, proxy, backends, history)
            run.record('plan', master_seed=self.seed, scenario=plan['scenario'], writer_seeds=plan['writer_seeds'],
                       clients=clients, rotations=rotations)
            await run.join([await self.start(address) for _ in plan['writers']])
            for rotation in range(1, rotations + 1):
                fields = await run.rotate(rotation, gated=rotation % 50 == 0 or rotation == rotations)
                self.assertEqual(self.state.attempts['CreateLayer'], 1)
                self.assertEqual(self.state.attempts.get('UpdateLayer', 0), clients * rotation)
                self.assertEqual(self.state.attempts['SubscribeReadLayer'], clients * (1 + rotation))
                run.record('rotation', rotation=rotation, commits=len(self.commits()), fields=run.fields_digest(),
                           **{key: fields[key] for key in ('backend', 'gated', 'retired', 'boundary', 'delay_ms')},
                           generations=fields['generations'])

            fresh = await self.start(address)
            state = await self.wait_fields(fresh, run.expected)
            self.assertEqual(state['pending'], 0, state)
            self.assertEqual(await client_update_attempts(fresh), 0)
            run.record('fresh-observer-converged', fields=len(run.expected))
            self.assertEqual(self.state.attempts['CreateLayer'], 1)
            self.assertEqual(self.state.attempts['SubscribeReadLayer'], clients * (1 + rotations) + 1)
            closed = await writer_tasks(*(self.close(driver) for driver in run.drivers + [fresh]))
            self.assertTrue(all(result['fields'] == run.expected for result in closed))
            boundary = 1 + clients * rotations
            self.assertEqual(len(self.commits()), boundary, 'teardown issued another mutation')
            self.assertEqual(run.expected, self.oracle())
            run.record('complete', rotations=rotations, admitted=clients * rotations, boundary=boundary)

    async def test_seeded_writers_soak_with_refresh_ping_and_bounded_growth(self):
        """Duration qualification: count × READY pause, exact accounting, bounded resources.

        The soak proves backend rotation and reconnect, proactive token refresh
        over wall time, server-side expiry with rejection recovery, a Ping per
        read generation and retained work that returns to its warm-up values.
        The pause is load (token lifetime, holdback of paced peer edits); nothing
        waits on it. Every rotation cuts every connection within about 5 s of
        READY, so periodic Ping estimation is covered only by the campaign's
        hold rotations, which wait in one READY generation for a second Ping,
        and gRPC keepalive is not exercised. Server-side expiry every
        expiry_every rotations alternates the read reconnect (odd events,
        expired before the outage) and the update reissue (even events, expired
        once the rotation converged, with one extra edit each). The scale comes
        from writer_soak: the campaign case sets campaign.soak_scale(); plain
        discovery uses campaign.SOAK_HARNESS, which runs every event and
        baseline path in 8 rotations and asserts the memory thresholds only at
        campaign scale.
        """
        clients, rotations = getattr(self, 'writer_clients', 4), getattr(self, 'writer_steps', 8)
        scale = getattr(self, 'writer_soak', campaign.SOAK_HARNESS)
        expiry_every, hold_every, warmup = scale['expiry_every'], scale['ping_hold_every'], scale['warmup']
        plan = rotation_writer_plan(self.seed, clients, rotations)
        tokens = {}
        for index in range(clients):
            for generation in range(1, campaign.SOAK_MAX_GENERATIONS + 1):
                self.state.issue(f'soak-w{index}-{generation}', 'writer')
                tokens[hashlib.sha256(f'soak-w{index}-{generation}'.encode()).hexdigest()] = (index, generation)
        prefix, proxy, backends, address = await self.rotation_setup(plan, 'soak-writers')
        started = time.monotonic()
        peers, extra, update_events, read_events, holds = campaign.SOAK_PEER_EDITS, 0, 0, 0, 0
        auth_log, current = [[] for _ in range(clients)], [1] * clients
        remembered, baseline, samples, series = [], None, [], [[] for _ in range(clients)]

        def rejections(method):
            """Rejected attempts per writer for one method, each naming a remembered token."""
            counts = [0] * clients
            for rejection in self.state.auth_rejections:
                if rejection['method'] == method:
                    counts[tokens[rejection['token_sha256']][0]] += 1
            return counts

        with Path(str(prefix) + '.history.jsonl').open('w') as history:
            run = RotationRun(self, plan, proxy, backends, history)

            async def auth_events(index):
                reply = await run.drivers[index].command('auth-events')
                auth_log[index].extend(reply['events'])
                current[index] = reply['generation']
                return reply

            async def expire_current():
                for index in range(clients):
                    await auth_events(index)
                    self.state.expire(f'soak-w{index}-{current[index]}')
                    remembered.append((index, current[index]))

            run.record('plan', master_seed=self.seed, scenario='public-plugin.soak.v1', writer_seeds=plan['writer_seeds'],
                       clients=clients, rotations=rotations, scale=scale, token_ttl_ms=campaign.SOAK_TOKEN_TTL_MS,
                       peer_edits=peers)
            await run.join([await self.start(address, environment=dict(
                LIVE_TEST_TOKEN_PREFIX=f'soak-w{index}', LIVE_TEST_TOKEN_TTL_MS=str(campaign.SOAK_TOKEN_TTL_MS)))
                for index in range(clients)])
            for rotation in range(1, rotations + 1):
                event = rotation % expiry_every == 0
                ordinal = rotation // expiry_every
                read_event, update_event = event and ordinal % 2 == 1, event and ordinal % 2 == 0
                # 1. A read event expires every writer's current token before the outage.
                if read_event:
                    await expire_current()
                    read_events += 1
                pings = [ping_results((await driver.command('diagnostics'))['snapshot']).get(0, 0)
                         for driver in run.drivers]
                # 2. The outage and recovery; a rejected reconnect adds retries. A
                # read event holds no caught-up reply: its rejected attempts
                # would take the gated attempt numbers.
                client_updates = [rotation + update_events + count for count in rejections('UpdateLayer')]
                fields = await run.rotate(rotation, gated=(rotation % 50 == 0 or rotation == rotations) and not read_event,
                                          relaxed=read_event, client_updates=client_updates)
                # 3. Every read generation pings: each writer finishes a Ping OK
                # after its reconnect, under a failure-only watchdog.
                for index, driver in enumerate(run.drivers):
                    async def pinged():
                        while True:
                            count = ping_results((await driver.command('diagnostics'))['snapshot']).get(0, 0)
                            if count > pings[index]:
                                return count
                    pings[index] = await asyncio.wait_for(pinged(), 15)
                # 4. The READY pause with paced peer commits: load, not synchronization.
                for ordinal_edit in range(peers):
                    edit = {f'peer_{(2 * rotation + ordinal_edit) % 3}': -(10000 * rotation + ordinal_edit)}
                    boundary = len(self.commits())
                    await self.peer(edit)
                    commit = self.commits()[boundary]
                    self.assertEqual((len(self.commits()), commit.writer, decode_update(commit.payload)),
                                     (boundary + 1, 'independent-peer', edit))
                    run.model.apply(commit.payload, commit.sequence)
                    run.expected.update(edit)
                    await asyncio.sleep(scale['pause_ms'] / 1000 / peers)
                # A hold rotation stays in its READY generation until every writer
                # finishes a periodic Ping OK, under a failure-only watchdog above
                # the runtime's 10 s Ping interval. The next rotation's exact
                # read-record check shows no reconnect happened meanwhile.
                hold = bool(hold_every) and rotation % hold_every == hold_every // 2
                if hold:
                    for index, driver in enumerate(run.drivers):
                        async def periodic():
                            while True:
                                count = ping_results((await driver.command('diagnostics'))['snapshot']).get(0, 0)
                                if count > pings[index]:
                                    return count
                        pings[index] = await asyncio.wait_for(periodic(), 20)
                    holds += 1
                # 5. Convergence: nothing pending, the peer commits applied, every writer converged.
                saves = await writer_tasks(*(driver.command('save 5000') for driver in run.drivers))
                self.assertTrue(all(result['result'] == 0 for result in saves), saves)
                self.assertEqual(run.expected, run.model.state)
                for driver in run.drivers:
                    self.assertEqual((await self.wait_fields(driver, run.expected))['pending'], 0)
                # 6. An update event expires every writer's current token after
                # convergence; each writer's extra edit is issued, rejected at
                # most once and reissued at most once (step 7 names each
                # rejection or the proactive refresh that preempted it).
                if update_event:
                    await expire_current()
                    update_events += 1
                    boundary = len(self.commits())
                    edits = [{f'event_w{index}': -(1000 * (index + 1) + ordinal)} for index in range(clients)]
                    for index, driver in enumerate(run.drivers):
                        name, value = next(iter(edits[index].items()))
                        self.assertEqual((await driver.command(f'usd-set 1 {name} {value}'))['result'], 0)
                    saves = await writer_tasks(*(driver.command('save 5000') for driver in run.drivers))
                    self.assertTrue(all(result['result'] == 0 for result in saves), saves)
                    extra += clients
                    commits = self.commits()[boundary:]
                    self.assertEqual(len(commits), clients)
                    for commit in commits:
                        index = next(i for i, identity in run.identities.items() if identity == commit.writer)
                        self.assertEqual(decode_update(commit.payload), edits[index])
                        run.model.apply(commit.payload, commit.sequence)
                        run.expected.update(edits[index])
                    for index, driver in enumerate(run.drivers):
                        self.assertEqual((await self.wait_fields(driver, run.expected))['pending'], 0)
                        run.status[index] = await driver.command('await-status 2')
                # 7. Exact accounting, adjusted by the recorded expiry events.
                read_rejected, update_rejected = sum(rejections('SubscribeReadLayer')), sum(rejections('UpdateLayer'))
                self.assertEqual(self.state.attempts['CreateLayer'], 1)
                self.assertEqual(self.state.attempts['UpdateLayer'],
                                 clients * rotation + peers * rotation + extra + update_rejected)
                self.assertEqual(self.state.attempts['SubscribeReadLayer'], clients * (1 + rotation) + read_rejected)
                self.assertEqual(len(self.commits()), 1 + clients * rotation + peers * rotation + extra)
                named = [tokens[rejection['token_sha256']] for rejection in self.state.auth_rejections]
                self.assertTrue(all(rejection['reason'] == 'expired' for rejection in self.state.auth_rejections),
                                self.state.auth_rejections)
                self.assertEqual(len(named), len(set(named)), named)
                self.assertLessEqual(set(named), set(remembered))
                for index in range(clients):
                    await auth_events(index)
                    challenges = [entry[0] for entry in auth_log[index]]
                    self.assertEqual(len(challenges), len(set(challenges)), auth_log[index])
                    self.assertTrue(all(entry[5] in (0, 15) for entry in auth_log[index]), auth_log[index])
                # Each expired generation was rejected by the service, or a
                # proactive EXPIRING refresh replaced it before any RPC used it.
                for index, generation in remembered:
                    self.assertTrue((index, generation) in named or any(
                        entry[2] == generation and entry[1] == 2 for entry in auth_log[index]),
                        (index, generation, named, auth_log[index]))
                # 8. No auth failure, every credential wait resolved usable, Ping
                # attempts end OK, UNAVAILABLE or CANCELLED only.
                snapshots = [await idle_executor_snapshot(driver) for driver in run.drivers]
                for snapshot in snapshots:
                    auth = snapshot['metrics']['auth']
                    self.assertEqual(auth['failures_by_cause'], [], auth)
                    self.assertTrue(all(entry['resolution'] == 0 for entry in auth['credential_wait_ms_by_resolution']['entries']),
                                    auth)
                    self.assertLessEqual(set(ping_results(snapshot)), {0, 8, 10}, ping_results(snapshot))
                # 9. Process figures and retained-work gauges.
                samples = [soak_sample(driver.process.pid, snapshot) for driver, snapshot in zip(run.drivers, snapshots)]
                for index, sample in enumerate(samples):
                    series[index].append((rotation, sample['anon']))
                # 10. After warm-up every gauge returns to its baseline value. The
                # baseline is the first rotation at or after warm-up without an
                # expiry event, so no event's transient state is in it.
                if baseline is None and rotation >= warmup and not event:
                    baseline = samples
                drift = [soak_drift(base, sample) for base, sample in zip(baseline, samples)] if baseline else []
                run.record('rotation', rotation=rotation, commits=len(self.commits()), fields=run.fields_digest(),
                           **{key: fields[key] for key in ('backend', 'gated', 'retired', 'boundary', 'delay_ms')},
                           generations=fields['generations'], expiry='read' if read_event else 'update' if update_event else None,
                           auth_generations=current, pings=pings, hold=hold, rejected=[read_rejected, update_rejected],
                           samples=samples, drift=drift)
                if baseline:
                    for index, (base, sample) in enumerate(zip(baseline, samples)):
                        self.assertEqual(drift[index], {}, (rotation, index))
                        self.assertLessEqual(sample['threads'], base['threads'] + campaign.SOAK_EXTRA_THREADS,
                                             (rotation, index, base['threads'], sample['threads']))
                        self.assertLessEqual(sample.get('timer_threads', 0), campaign.SOAK_MAX_GRPC_TIMER_THREADS,
                                             (rotation, index, sample.get('timer_threads')))
                        self.assertLessEqual(sample['files'], base['files'] + campaign.SOAK_EXTRA_FILES,
                                             (rotation, index, base['files'], sample['files']))

            slopes = [rss_slope([point for point in points if point[0] >= rotations // 2]) for points in series]
            challenges = [{reason: sum(entry[1] == reason for entry in log) for reason in (1, 2, 3)} for log in auth_log]
            summary = dict(
                schema_version=1, master_seed=self.seed, clients=clients, rotations=rotations,
                wall_time_s=round(time.monotonic() - started, 3), scale=scale, holds=holds,
                read_events=read_events, update_events=update_events, extra_commits=extra,
                rejections=dict(read=sum(rejections('SubscribeReadLayer')), update=sum(rejections('UpdateLayer'))),
                commits=len(self.commits()), attempts=dict(self.state.attempts),
                challenges=[{'missing': c[1], 'expiring': c[2], 'rejected': c[3]} for c in challenges],
                generations=current, pings_ok=pings, read_generations=run.generations,
                baseline=baseline, final=samples, anon_slope_bytes_per_rotation=slopes,
                thresholds=dict(rss_ratio=campaign.SOAK_RSS_RATIO, rss_slack_bytes=campaign.SOAK_RSS_SLACK_BYTES,
                                rss_slope_bytes=campaign.SOAK_RSS_SLOPE_BYTES, extra_threads=campaign.SOAK_EXTRA_THREADS,
                                max_grpc_timer_threads=campaign.SOAK_MAX_GRPC_TIMER_THREADS,
                                extra_files=campaign.SOAK_EXTRA_FILES, max_generations=campaign.SOAK_MAX_GENERATIONS))
            Path(str(prefix) + '.soak-summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
            print('soak-summary ' + json.dumps(dict(rotations=rotations, wall_time_s=summary['wall_time_s'],
                                                   anon=[sample['anon'] for sample in samples], slopes=slopes)), flush=True)
            self.assertTrue(all(generation < campaign.SOAK_MAX_GENERATIONS for generation in current), current)
            self.assertIsNotNone(baseline)
            self.assertEqual((read_events, update_events), (rotations // expiry_every - rotations // expiry_every // 2,
                                                            rotations // expiry_every // 2))
            if scale['thresholds']:
                # The runtime's own clock refreshed every writer's token at least once.
                self.assertTrue(all(counts[2] >= 1 for counts in challenges), challenges)
                self.assertEqual(holds, (rotations + hold_every // 2) // hold_every)
                bound = lambda value: campaign.SOAK_RSS_RATIO * value + campaign.SOAK_RSS_SLACK_BYTES
                for index, (base, sample) in enumerate(zip(baseline, samples)):
                    self.assertLessEqual(sample['anon'], bound(base['anon']), (index, base['anon'], sample['anon']))
                    # VmHWM never falls, so reclaim cannot hide a transient peak.
                    self.assertLessEqual(sample['hwm'], bound(base['hwm']), (index, base['hwm'], sample['hwm']))
                    self.assertLess(slopes[index], campaign.SOAK_RSS_SLOPE_BYTES, (index, slopes[index]))

            fresh = await self.start(address)
            state = await self.wait_fields(fresh, run.expected)
            self.assertEqual(state['pending'], 0, state)
            self.assertEqual(await client_update_attempts(fresh), 0)
            run.record('fresh-observer-converged', fields=len(run.expected))
            self.assertEqual(self.state.attempts['SubscribeReadLayer'],
                             clients * (1 + rotations) + sum(rejections('SubscribeReadLayer')) + 1)
            closed = await writer_tasks(*(self.close(driver) for driver in run.drivers + [fresh]))
            self.assertTrue(all(result['fields'] == run.expected for result in closed))
            boundary = 1 + clients * rotations + peers * rotations + extra
            self.assertEqual(len(self.commits()), boundary, 'teardown issued another mutation')
            self.assertEqual(run.expected, self.oracle())
            run.record('complete', rotations=rotations, boundary=boundary, wall_time_s=summary['wall_time_s'])

    async def test_rotation_before_execution_retains_uncertain_and_queued_writer_intents(self):
        await self.uncertain_writer_rotation('before')

    async def test_rotation_after_commit_retains_uncertain_and_queued_writer_intents(self):
        await self.uncertain_writer_rotation('after_commit')

    async def uncertain_writer_rotation(self, boundary):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        plan = writer_plan(self.seed, clients=4, steps=4)
        from tools.live.writer_campaign import observe_workload
        observe_workload(self, plan)
        plan.update(scenario='public-plugin.uncertain-rotation.v1', fault_boundary=boundary,
                    initial_fixture_sha256=hashlib.sha256(empty_snapshot()).hexdigest(),
                    candidate=json.loads(Path(os.environ['LIVE_TEST_PLUGIN_DRIVER'] + '.manifest.json').read_text()))
        prefix = self.artifact / self.id().split('.')[-1]
        Path(str(prefix) + '.plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
        print(f"uncertain-writers master_seed={self.seed} boundary={boundary} plan={prefix}.plan.json", flush=True)
        self.trace['writer_plan'] = plan
        backend = ReferenceServer(self.state, 'B', service_factory=scene_support.CompactSceneService)
        self.extra_servers.append(backend)
        other = await backend.start()
        proxy = scene_support.RotatingProxy(self.address)
        self.proxies.append(proxy)
        address = await proxy.start()
        admissions = [[] for _ in plan['writers']]
        admitted = [0] * len(plan['writers'])
        clients = []
        before_cut, frozen = [], []
        peer_write = {'shared_peer': 9000001}
        with Path(str(prefix) + '.history.jsonl').open('w') as history:
            def record(kind, **fields):
                history.write(json.dumps(dict(event=kind, **fields), sort_keys=True) + '\n')
                history.flush()

            async def attempts(driver, index, stage):
                metrics = (await driver.command('diagnostics'))['snapshot']['metrics']
                count = sum(item['count'] for item in metrics['rpc_attempts_by_operation'] if item['operation'] == 4)
                record('client-update-attempts', writer=index, stage=stage, count=count)
                self.assertEqual(count, 1, (index, stage, metrics))

            record('plan', master_seed=self.seed, boundary=boundary, writer_seeds=plan['writer_seeds'])
            drivers = [await self.start(address) for _ in plan['writers']]
            identities = [await driver.command('observe-status') for driver in drivers]
            # An unfaulted peer commit installed by every writer before any
            # local edit puts the before-cut floor at least one ledger entry up.
            await self.peer(peer_write)
            for driver in drivers:
                await self.wait_fields(driver, peer_write)
            record('peer-installed', fields=peer_write)
            base = self.state.attempts['UpdateLayer']
            faults = [self.state.plan('UpdateLayer', boundary, attempt=base + i + 1, pause=True) for i in range(4)]
            try:
                async def author(index, driver):
                    for op in plan['writers'][index]:
                        record('submit', **op)
                        # One plain USD change block; its flush job has run when
                        # the command returns. Every first RPC is held, so no
                        # admitted intent retires.
                        result = await driver.command(f"usd-set-admit 1 {op['field']} {op['value']}")
                        self.assertEqual((result['result'], result['status_result']), (0, 0), result)
                        admitted[index] += 1
                        self.assertEqual(result['pending'], admitted[index], result)
                        entry = dict(edit=op['edit'], ordinal=admitted[index])
                        admissions[index].append(entry)
                        record('admitted', writer=index, **entry)
                async with asyncio.TaskGroup() as authors:
                    for index, driver in enumerate(drivers):
                        authors.create_task(author(index, driver))
                await asyncio.wait_for(asyncio.gather(*(fault.reached.wait() for fault in faults)), 5)
                for index, driver in enumerate(drivers):
                    await attempts(driver, index, 'first-rpc-held')
                    state = await driver.command('state')
                    self.assertEqual(state['pending'], 4)
                    self.assertEqual(state['fields']['shared_peer'], peer_write['shared_peer'])
                    record('before-cut', writer=index, state=state)
                    before_cut.append(state['fields'])
                self.assertEqual(self.state.attempts['UpdateLayer'], base + 4)
                committed = 4 if boundary == 'after_commit' else 0
                self.assertEqual(len(self.commits()), committed + 2)
                retired = await asyncio.wait_for(proxy.rotate_and_wait(other, pause=True), 5)
                self.assertGreaterEqual(retired, 4)
                # Transport retirement alone cannot prove the old service handler
                # stopped before execution. Each paused handler must see actual
                # cancellation while its release event remains unset.
                await asyncio.wait_for(asyncio.gather(*(fault.cancelled.wait() for fault in faults)), 5)
                self.assertTrue(all(not fault.release.is_set() for fault in faults))
                record('cut-and-service-cancelled', backend='B', retired_forwards=retired,
                       cancelled_attempts=[fault.attempt for fault in faults], committed=committed)
                self.assertEqual(len(self.commits()), committed + 2)
                for index, driver in enumerate(drivers):
                    status = await driver.command('await-status 7')
                    self.assertEqual(status['subject'], identities[index]['subject'])
                    self.assertGreater(status['incident'], 0)
                    state = await driver.command('state')
                    self.assertEqual((state['state'], state['result'], state['pending']), (7, 19, 4))
                    record('uncertain', writer=index, status=status, state=state)
                    frozen.append(state['fields'])
                    await attempts(driver, index, 'terminal')
                    clients.append(dict(status=status, before_cut=before_cut[index]))
            finally:
                for fault in faults:
                    fault.release.set()
                proxy.gate.set()

            commit_records = [dict(sequence=entry.sequence, writer=entry.writer, wire=entry.payload.hex())
                              for entry in self.commits()[1:]]
            planned_first = {op[0]['field']: op[0]['value'] for op in plan['writers']}
            self.assertEqual(self.oracle(), dict(peer_write, **planned_first) if committed else peer_write)
            # Actual commit bytes account for precisely the first operation from
            # every writer; queued suffixes must never appear on the service.
            if committed:
                partial = dict(plan, writers=[ops[:1] for ops in plan['writers']])
                fields, _ = account_writers(partial, [items[0] for items in admissions], [
                    dict(sequence=entry.sequence, writer=entry.writer, fields=decode_update(entry.payload))
                    for entry in self.commits()[2:]], base_sequence=self.commits()[1].sequence)
                self.assertEqual(fields, planned_first)
            for index, (driver, expected) in enumerate(zip(drivers, clients)):
                self.assertEqual((await driver.command('save 5000'))['result'], 19)
                path = Path(str(prefix) + f'.writer-{index}.usda')
                exported = await driver.command(f'export {path}')
                self.assertEqual(exported['result'], 0, exported)
                state = await driver.command('state')
                # FAILED latched at await-status 7, so the host content is
                # already frozen there; SdfLayer::Export writes exactly it.
                self.assertEqual(state['fields'], frozen[index])
                self.assertEqual(exported['fields'], state['fields'])
                expected.update(export=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), state=state)
                await attempts(driver, index, 'exported')
                record('exported', writer=index, **expected)
            self.assertEqual(self.state.attempts['UpdateLayer'], base + 4)
            evidence = dict(plan=plan, clients=clients, commits=commit_records, peers=[peer_write])
            evidence_path = Path(str(prefix) + '.offline-input.json')
            evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
            closed = await asyncio.gather(*(self.close(driver) for driver in drivers))
            self.assertEqual([result['fields'] for result in closed], [entry['state']['fields'] for entry in clients])
            self.assertTrue(all(driver.process.returncode == 0 for driver in drivers))
            # A new interpreter sees only persisted synthetic evidence and the
            # exported files, after every original native process has exited.
            process = await asyncio.create_subprocess_exec(sys.executable, '-B',
                str(Path(__file__).with_name('writer_recovery.py')), str(evidence_path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                output, _ = await asyncio.wait_for(process.communicate(), 8)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            Path(str(prefix) + '.offline-report.log').write_bytes(output)
            self.assertEqual(process.returncode, 0, output.decode(errors='replace'))
            report = json.loads(output)
            self.assertEqual(report['verified_operations'], 16)
            record('offline-verified', report=report)
            fresh = await self.start(address)
            expected = dict(peer_write, **planned_first) if committed else peer_write
            await self.wait_fields(fresh, expected)
            self.assertEqual(self.state.scene_reads[-1]['backend'], 'B')
            self.assertEqual((await self.close(fresh))['fields'], expected)
            self.assertEqual(self.state.attempts['UpdateLayer'], base + 4)
            self.assertEqual(self.state.attempts['CreateLayer'], 1)
            self.assertEqual(len(self.commits()), committed + 2)
            record('complete', admitted=16, uncertain=4, queued=12, committed=committed)

    async def test_packaged_plugin_unknown_write_exports_the_failed_layer(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        driver = await self.start()
        fault = self.state.plan("UpdateLayer", "after_commit", pause=True, code=grpc.StatusCode.UNAVAILABLE)
        authored = await driver.command("usd-set 1 uncertain 71")
        self.assertEqual((authored["result"], authored["status_result"], authored["pending"]), (0, 0, 1), authored)
        await asyncio.wait_for(fault.reached.wait(), 5)
        fault.release.set()
        failed = await driver.command("failed")
        self.assertEqual((failed["result"], failed["state"], failed["pending"]), (19, 7, 1))
        self.assertEqual((await driver.command("save 1000"))["result"], 19)
        path = self.artifact / (self.id().split(".")[-1] + ".usda")
        attempts = dict(self.state.attempts)
        exported = await driver.command(f"export {path}")
        self.assertEqual(exported, {"type": "exported", "result": 0, "fields": {"uncertain": 71}})
        original = path.read_bytes()
        self.assertTrue(original.startswith(b"#usda 1.0"))
        self.assertEqual(self.state.attempts, attempts, "SdfLayer::Export issued a service operation")
        self.assertEqual([decode_update(c.payload) for c in self.commits()[1:]], [{"uncertain": 71}])
        self.assertEqual((await driver.command("state"))["pending"], 1)
        self.assertEqual((await self.close(driver))["fields"], {"uncertain": 71})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)


class PluginGraphReadTests(unittest.IsolatedAsyncioTestCase):
    """Root-field reads of the packaged plugin against the independent graph service.

    Companion to PluginTests.test_reopen_of_a_released_layer_stats_again; the root-field compact service has no
    unverified catch-up.
    """
    graph_creation = True
    asyncSetUp = PluginTests.asyncSetUp
    cleanup = PluginTests.cleanup
    commits = PluginTests.commits
    peer = PluginTests.peer
    start = PluginTests.start
    wait_fields = PluginTests.wait_fields
    close = PluginTests.close

    def oracle(self):
        root = next(node for node in self.state.sessions[self.session].layers["root.live"].model.state
                    if node["id"] == 1)
        return {key: value for key, value in root["fields"].items() if type(value) is int}

    async def test_reopen_with_an_unverified_catch_up_applies_own_and_peer_commits(self):
        # The reopened layer reads with the endpoint's writer ID. Without a proven stable cut its snapshot is acked
        # below the peer and own commits it holds, which are delivered again, the own one included, so the layer
        # converges on the service's state.
        await self.peer({"start": 1})
        driver = await self.start()
        self.assertEqual((await driver.command("usd-set 1 start 2"))["result"], 0)
        self.assertEqual((await driver.command("save 5000"))["result"], 0)
        await self.wait_fields(driver, {"start": 2})
        await self.peer({"other": 5})
        await self.wait_fields(driver, {"start": 2, "other": 5})
        self.assertEqual([(commit.sequence, commit.writer == "independent-peer") for commit in self.commits()],
                         [(1, False), (2, True), (3, False), (4, True)])
        self.state.unverified_cut = 3  # Acks the snapshot at 1 and delivers 2, 3 and 4 again.
        reopened = await driver.command("reopen-root")
        self.assertEqual((reopened["type"], reopened["result"], reopened["state"]), ("reopened", 0, 2), reopened)
        self.assertEqual(self.state.reads[-1]["cursor"], 0)
        expected = {"start": 2, "other": 5}
        self.assertEqual(self.oracle(), expected)
        # Once the layer has applied every delivered position through the service's latest, its host layer holds
        # their effects.
        caught_up = await driver.command("await-sequences 4 4")
        self.assertEqual((caught_up["result"], caught_up["state"], caught_up["fields"]), (0, 2, expected), caught_up)
        await self.peer({"end": 9})
        expected["end"] = 9
        self.assertEqual(self.oracle(), expected)
        settled = await driver.command("await-sequences 5 5")
        self.assertEqual((settled["result"], settled["state"], settled["fields"]), (0, 2, expected), settled)
        self.assertEqual((await self.close(driver))["fields"], expected)


class PluginStructuralAuthoringTests(unittest.IsolatedAsyncioTestCase):
    """Plain USD namespace authoring against the independent graph service.

    Companion to PluginTests.test_direct_usd_authoring_batches_change_blocks_and_converges;
    the root-field compact service cannot carry structural commands.
    """
    graph_creation = True
    asyncSetUp = PluginTests.asyncSetUp
    cleanup = PluginTests.cleanup
    commits = PluginTests.commits
    start = PluginTests.start
    close = PluginTests.close

    def model(self):
        return self.state.sessions[self.session].layers["root.live"].model

    def oracle(self):
        root = next(node for node in self.model().state if node["id"] == 1)
        return {key: value for key, value in root["fields"].items() if type(value) is int}

    def prim_oracle(self):
        nodes = self.model().state
        paths = {node["id"]: node["path"] for node in nodes}
        return {"paths": sorted(node["path"] for node in nodes if node["type"] == graph_model.PRIM),
                "values": {paths[node["parent"]]: node["fields"]["default"] for node in nodes
                           if node["type"] == graph_model.ATTRIBUTE and node["name"] == "value"
                           and type(node["fields"].get("default")) is int}}

    async def wait_prims(self, driver, expected):
        command = "wait-prims " + str(len(expected["paths"])) + "".join(
            f" {path} {expected['values'][path]}" for path in expected["paths"])
        result = await driver.command(command)
        self.assertEqual(result["result"], 0, result)
        self.assertEqual({"paths": result["paths"], "values": result["values"]}, expected)
        return result

    async def converged(self, drivers, expected):
        self.assertEqual(self.prim_oracle(), expected)
        for driver in drivers:
            await self.wait_prims(driver, expected)
        fresh = await self.start()
        await self.wait_prims(fresh, expected)
        await self.close(fresh)

    async def committed(self, count):
        async def settled():
            async with self.state.changed:
                await self.state.changed.wait_for(lambda: len(self.commits()) >= count)
        try:
            await asyncio.wait_for(settled(), 5)
        except TimeoutError:
            self.fail(f"service committed {len(self.commits())} of {count}; rejections={self.state.rejections}")

    async def test_direct_usd_structural_authoring_converges(self):
        first, second = await self.start(), await self.start()
        second_read = self.state.reads[-1]["attempt"]
        drivers = (first, second)
        # (a) Define a prim and its attribute in one change block: one UpdateLayer.
        updates = self.state.attempts.get("UpdateLayer", 0)
        self.assertEqual((await first.command("usd-prim define P 1"))["result"], 0)
        self.assertEqual((await first.command("save 5000"))["result"], 0)
        self.assertEqual(self.state.attempts.get("UpdateLayer", 0), updates + 1)
        await self.converged(drivers, {"paths": ["/P"], "values": {"/P": 1}})
        # (b) A renames P while B, not yet holding the rename, edits /P.value.
        gate = self.state.plan_read_frame(second_read, self.commits()[-1].sequence + 1)
        self.addCleanup(gate.release.set)
        self.assertEqual((await first.command("usd-prim rename P Q"))["result"], 0)
        self.assertEqual((await first.command("save 5000"))["result"], 0)
        await asyncio.wait_for(gate.reached.wait(), 5)
        self.assertEqual((await second.command("usd-prim set P 2"))["result"], 0)
        await self.committed(len(self.commits()) + 1)
        gate.release.set()
        self.assertEqual((await second.command("save 5000"))["result"], 0)
        await self.converged(drivers, {"paths": ["/Q"], "values": {"/Q": 2}})
        # (d) Create and remove a prim in one host turn: the layer stays healthy and issues nothing.
        updates = self.state.attempts.get("UpdateLayer", 0)
        self.assertEqual((await first.command("usd-prim create-remove T 7"))["result"], 0)
        self.assertEqual((await first.command("save 5000"))["result"], 0)
        state = await first.command("state")
        self.assertEqual((state["result"], state["pending"]), (0, 0), state)
        self.assertEqual(self.state.attempts.get("UpdateLayer", 0), updates)  # An empty compiled envelope issues nothing.
        await self.converged(drivers, {"paths": ["/Q"], "values": {"/Q": 2}})
        self.assertFalse(self.state.rejections)
        for driver in drivers:
            self.assertEqual((await self.close(driver))["fields"], {})

    async def concurrent_delete_and_edit(self):
        """A deletes P while B, not yet holding the delete, edits /P.value and saves."""
        first, second = await self.start(), await self.start()
        second_read = self.state.reads[-1]["attempt"]
        drivers = (first, second)
        self.assertEqual((await first.command("usd-prim define P 1"))["result"], 0)
        self.assertEqual((await first.command("save 5000"))["result"], 0)
        await self.converged(drivers, {"paths": ["/P"], "values": {"/P": 1}})
        gate = self.state.plan_read_frame(second_read, self.commits()[-1].sequence + 1)
        self.addCleanup(gate.release.set)
        self.assertEqual((await first.command("usd-prim remove P"))["result"], 0)
        self.assertEqual((await first.command("save 5000"))["result"], 0)
        await asyncio.wait_for(gate.reached.wait(), 5)
        self.assertEqual((await second.command("usd-prim set P 3"))["result"], 0)
        await self.committed(len(self.commits()) + 1)
        # The last commit is B's edit alone: its /P.value default of 3.
        edit = graph_decode(self.commits()[-1].payload)
        self.assertEqual([(field.key, field.value) for field in edit.fields], [("default", 3)], edit)
        gate.release.set()
        updates, count = self.state.attempts.get("UpdateLayer", 0), len(self.commits())
        saved = await second.command("save 5000")
        state = await second.command("state")
        self.trace["states"] += [saved, state]
        return drivers, saved, state, updates, count

    async def test_direct_usd_concurrent_delete_and_edit_reaches_a_known_outcome(self):
        drivers, saved, state, updates, count = await self.concurrent_delete_and_edit()
        issued = self.state.attempts.get("UpdateLayer", 0) - updates
        observed = dict(result=saved["result"], state=state["state"], pending=state["pending"])
        self.assertEqual((observed, issued, len(self.commits())), (dict(result=0, state=2, pending=0), 0, count), state)
        self.assertNotEqual(self.commits()[-1].writer, self.commits()[-2].writer)
        await self.converged(drivers, {"paths": [], "values": {}})
        self.assertFalse(self.state.rejections)

    # A's delete commits first (sequence 3); B, whose read is held before that frame, issues its
    # edit and the service commits it as orphan data (sequence 4). Supersession happens when B
    # replays the peer delete.
    async def test_direct_usd_edit_of_concurrently_deleted_prim_is_superseded(self):
        drivers, saved, state, updates, count = await self.concurrent_delete_and_edit()
        # The service commits B's edit of the deleted attribute as orphan data.
        self.assertEqual((saved["result"], state["result"], state["state"], state["pending"]), (0, 0, 2, 0), state)
        self.assertEqual([entry.sequence for entry in self.commits()[-2:]], [3, 4])
        self.assertEqual((self.state.attempts.get("UpdateLayer", 0), len(self.commits())), (updates, count))
        await self.converged(drivers, {"paths": [], "values": {}})
        self.assertFalse(self.state.rejections)
        for driver in drivers:
            self.assertEqual((await self.close(driver))["fields"], {})

    # B's flush job is withheld until B has applied A's delete, so B's edit of /P.value is still
    # unissued: the delete publication's own flush finds every record superseded and issues nothing.
    async def test_direct_usd_edit_superseded_before_issue_is_dropped_without_failure(self):
        first, second = await self.start(), await self.start()
        second_read = self.state.reads[-1]["attempt"]
        drivers = (first, second)
        self.assertEqual((await first.command("usd-prim define P 1"))["result"], 0)
        self.assertEqual((await first.command("save 5000"))["result"], 0)
        await self.converged(drivers, {"paths": ["/P"], "values": {"/P": 1}})
        gate = self.state.plan_read_frame(second_read, self.commits()[-1].sequence + 1)
        self.addCleanup(gate.release.set)
        self.assertEqual((await first.command("usd-prim remove P"))["result"], 0)
        self.assertEqual((await first.command("save 5000"))["result"], 0)
        await asyncio.wait_for(gate.reached.wait(), 5)
        await second.command("hold-dispatch")
        self.assertEqual((await second.command("usd-prim set P 3"))["result"], 0)
        updates, count = self.state.attempts.get("UpdateLayer", 0), len(self.commits())
        gate.release.set()
        applied = await second.command("wait-prims 0")
        self.assertEqual((applied["result"], applied["paths"]), (0, []), applied)
        released = await second.command("release-dispatch")
        state = await second.command("state")
        self.trace["states"] += [applied, released, state]
        self.assertEqual((state["result"], state["state"], state["pending"]), (0, 2, 0), state)
        self.assertEqual((self.state.attempts.get("UpdateLayer", 0), len(self.commits())), (updates, count))
        await self.converged(drivers, {"paths": [], "values": {}})
        self.assertFalse(self.state.rejections)
        for driver in drivers:
            self.assertEqual((await self.close(driver))["fields"], {})

    async def test_direct_usd_three_writers_conflict_and_a_fresh_observer_converge(self):
        """Three plain-USD writer processes and a fresh observer; every edit is accounted."""
        first = await self.start()
        second = await self.start()
        second_read = self.state.reads[-1]["attempt"]
        third = await self.start()
        third_read = self.state.reads[-1]["attempt"]
        drivers = (first, second, third)
        base, updates = len(self.commits()), self.state.attempts.get("UpdateLayer", 0)
        authored, values = [], {}

        def prims():
            return {"paths": sorted(values), "values": dict(values)}

        async def author(index, kind, name, value=None):
            command = f"usd-prim {kind} {name}" + ("" if value is None else f" {value}")
            self.assertEqual((await drivers[index].command(command))["result"], 0)
            authored.append(dict(edit=f"e{len(authored) + 1}", writer=index, kind=kind, prim="/" + name, value=value))

        async def save(index):
            self.assertEqual((await drivers[index].command("save 5000"))["result"], 0)

        async def settle():
            for driver in drivers:
                await self.wait_prims(driver, prims())

        # Distinct prims, one writer at a time; each converges before the next.
        for index, name, value in ((0, "Shared", 1), (0, "Doomed", 1), (0, "Alpha", 10), (1, "Beta", 20), (2, "Gamma", 30)):
            await author(index, "define", name, value)
            await save(index)
            values["/" + name] = value
            await settle()
        # Same field: B and C author /Shared.value before they receive A's value.
        count = len(self.commits())
        gates = [self.state.plan_read_frame(read, count + 1) for read in (second_read, third_read)]
        for gate in gates:
            self.addCleanup(gate.release.set)
        await author(0, "set", "Shared", 100)
        await save(0)
        await asyncio.wait_for(asyncio.gather(*(gate.reached.wait() for gate in gates)), 5)
        await author(1, "set", "Shared", 200)
        await self.committed(count + 2)
        await author(2, "set", "Shared", 300)
        await self.committed(count + 3)
        for gate in gates:
            gate.release.set()
        await save(1)
        await save(2)
        values["/Shared"] = 300  # Service order: C committed last.
        await settle()
        # Delete versus edit: A removes /Doomed while B, not yet holding the removal, edits it.
        count = len(self.commits())
        gate = self.state.plan_read_frame(second_read, count + 1)
        self.addCleanup(gate.release.set)
        await author(0, "remove", "Doomed")
        await save(0)
        await asyncio.wait_for(gate.reached.wait(), 5)
        await author(1, "set", "Doomed", 5)
        await self.committed(count + 2)  # The service keeps B's value as orphan data.
        gate.release.set()
        await save(1)
        del values["/Doomed"]
        await settle()
        # Every change block is one commit by its own writer, accounted reflected or superseded.
        scene, lineage = account_direct_prims(authored, self.commits()[base:])
        self.assertEqual(scene, prims())
        self.assertEqual(scene, self.prim_oracle())
        self.assertEqual({edit: entry.get('superseded_by', entry['outcome']) for edit, entry in lineage.items()},
                         {'e1': 'reflected', 'e2': 'e9', 'e3': 'reflected', 'e4': 'reflected', 'e5': 'reflected',
                          'e6': 'e7', 'e7': 'e8', 'e8': 'reflected', 'e9': 'reflected', 'e10': 'e9'})
        self.assertEqual(self.state.attempts.get("UpdateLayer", 0) - updates, len(authored))
        self.assertEqual([await client_update_attempts(driver) for driver in drivers], [5, 3, 2])
        # A fresh observer converges and issues nothing, through close.
        count, updates = len(self.commits()), self.state.attempts.get("UpdateLayer", 0)
        fresh = await self.start()
        await self.wait_prims(fresh, prims())
        self.assertEqual(await client_update_attempts(fresh), 0)
        self.assertEqual((await self.close(fresh))["fields"], {})
        self.assertEqual((len(self.commits()), self.state.attempts.get("UpdateLayer", 0)), (count, updates))
        self.assertFalse(self.state.rejections)
        self.trace["lineage"] = lineage
        for driver in drivers:
            self.assertEqual((await self.close(driver))["fields"], {})


class PluginCreationTests(unittest.IsolatedAsyncioTestCase):
    """Only installed public headers/shared plugin; independent graph service."""
    graph_creation = True
    asyncSetUp = PluginTests.asyncSetUp
    cleanup = PluginTests.cleanup
    commits = PluginTests.commits
    oracle = PluginTests.oracle
    start = PluginTests.start
    close = PluginTests.close

    async def prepared(self, command='creation-prepare'):
        driver = await self.start()
        before = dict(self.state.attempts)
        result = await driver.command(command)
        self.assertEqual((result['result'], result['phase'], result['started']), (0, 0, 0))
        self.assertGreater(result['subject'], 0)
        self.assertEqual(result['fields'], {})
        self.assertEqual(result['state'], 1)
        self.assertEqual(self.state.attempts, before, 'preparation/wrong-host calls issued an RPC')
        self.assertEqual((await driver.command('creation-wait 0'))['result'], 14)
        return driver, result['subject']

    async def test_import_of_own_path_after_a_failed_read_is_rejected(self):
        # An existing service layer the driver has not opened; its first read is refused.
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name="created.live")),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=empty_snapshot()))]), metadata=WRITER, timeout=5)
        driver = await self.start()
        self.state.plan("SubscribeReadLayer", "before", code=grpc.StatusCode.PERMISSION_DENIED)
        reads = self.state.attempts["SubscribeReadLayer"]
        imported = await driver.command('usd-new-import-twice')
        # The first Import fails its read and leaves the layer unattached; the second is
        # rejected UNSUPPORTED before any RPC instead of opening the layer.
        self.assertEqual((imported['first'], imported['first_result'], imported['second'], imported['second_result']),
                         (0, 7, 0, 12), imported)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], reads + 1)

    def created_commits(self):
        return [entry for entry in self.state.ledger if entry.layer == 'created.live']

    def assert_seed(self, counter=17, source_format='usda'):
        commits = self.created_commits()
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0].operation, 'CreateLayer')
        seed = graph_decode(commits[0].payload)
        self.assertTrue(seed.is_diff)
        self.assertEqual((seed.base_version, seed.source_format), (0, source_format))
        self.assertEqual([(field.section, field.key, field.value) for field in seed.fields], [(1, 'counter', counter)])

    def assert_empty_created_seed(self, *, count=1):
        entries = self.created_commits()
        self.assertEqual(len(entries), count)
        entry = entries[-1]
        self.assertEqual((entry.operation, entry.sequence), ('CreateLayer', 1))
        seed = graph_decode(entry.payload)
        self.assertEqual((seed.is_diff, seed.base_version, seed.source_format, seed.fields), (True, 0, '', ()))
        self.assertEqual([(item.tag, item.section, item.parent, item.name, item.spec_type, item.order,
                           item.old_parent, item.child_list, item.children) for item in seed.commands],
                         [(5, 1, 0, '', 7, 0, 0, 0, ())])
        return entry

    async def retained_prepared(self, command='creation-retained-prepare'):
        driver = await self.start()
        attempts = dict(self.state.attempts)
        prepared = await driver.command(command)
        self.assertEqual((prepared['result'], prepared['listed']), (0, 1))
        self.assertGreater(prepared['subject'], 0)
        self.assertEqual(prepared['uri'], 'live://' + self.address + '/' + quote(self.session, safe='') + '/created.live')
        for command in ('creation-retained-list', 'creation-retained-checks',
                        'creation-retained-replace-context', 'creation-retained-list', 'creation-retained-checks'):
            report = await driver.command(command)
            self.assertEqual(report, prepared, command)
        self.assertEqual(self.state.attempts, attempts,
                         'retained preparation/discovery/rejected calls issued service work')
        self.assertFalse(self.created_commits())
        return driver, prepared['subject'], attempts

    async def test_public_retained_creation_transfers_original_seed_after_context_owner_destruction(self):
        driver, subject, attempts = await self.retained_prepared()
        acquired = await driver.command('creation-retained-acquire')
        self.assertEqual((acquired['result'], acquired['subject'], acquired['phase'], acquired['started']),
                         (0, subject, 0, 0))
        self.assertEqual((acquired['state'], acquired['fields']), (1, {}))
        self.assertEqual((await driver.command('creation-retained-list'))['listed'], 0)
        self.assertEqual(self.state.attempts, attempts, 'ownership transfer issued an RPC')
        self.assertEqual((await driver.command('creation-start'))['result'], 0)
        completed = await driver.command('creation-wait 5000')
        self.assertEqual((completed['result'], completed['subject'], completed['phase'], completed['outcome']),
                         (0, subject, 2, 0))
        self.assert_seed()  # Source was changed to 99 and destroyed before transfer.
        self.assertEqual((await driver.command('creation-layer'))['result'], 0)
        activated = await driver.command('creation-activate')
        self.assertEqual((activated['result'], activated['fields'], activated['subject']),
                         (0, {'counter': 17}, subject))
        before_repeat = dict(self.state.attempts)
        self.assertEqual((await driver.command('creation-start'))['result'], 5)
        self.assertEqual((await driver.command('creation-activate'))['fields'], {'counter': 17})
        self.assertEqual(self.state.attempts, before_repeat)
        await self.close(driver)
        self.assert_seed()

    async def retained_live_seed_creation(self, command):
        await self.stub.UpdateLayer(iter([
            pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name='root.live',
                                                             object_id=1, client_id='peer')),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=update({'counter': 31})))]), metadata=WRITER, timeout=5)
        driver, subject, attempts = await self.retained_prepared(command)
        await self.stub.UpdateLayer(iter([
            pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name='root.live',
                                                             object_id=1, client_id='peer')),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=update({'counter': 72})))]), metadata=WRITER, timeout=5)
        advanced = await driver.command('wait 1 counter 72')
        self.assertEqual(advanced['fields'], {'counter': 72})
        self.assertEqual((await driver.command('creation-retained-acquire'))['subject'], subject)
        self.assertEqual((await driver.command('creation-start'))['result'], 0)
        self.assertEqual((await driver.command('creation-wait 5000'))['phase'], 2)
        self.assert_seed(31, '')
        self.assertEqual((await driver.command('creation-layer'))['result'], 0)
        active = await driver.command('creation-activate')
        self.assertEqual((active['result'], active['fields']), (0, {'counter': 31}))
        self.assertEqual((await driver.command('creation-start'))['result'], 5)
        await self.close(driver)
        self.assert_seed(31, '')

    async def test_public_retained_live_layer_seed_preserves_empty_source_format(self):
        await self.retained_live_seed_creation('creation-retained-prepare-live')

    async def test_public_retained_global_live_layer_seed_preserves_empty_source_format(self):
        await self.retained_live_seed_creation('creation-retained-prepare-live-global')

    async def test_public_retained_global_preparation_cleanup_never_issues_create(self):
        driver, subject, attempts = await self.retained_prepared('creation-retained-prepare-global')
        released = await driver.command('creation-retained-release')
        self.assertEqual(released['subject'], subject)
        self.assertIn(released['result'], (0, 14))
        if released['result'] == 14:
            # Explicit cleanup begins cancellation but retains a recoverable
            # owner until native callbacks retire. Transfer it, then await the
            # existing finite CV; no timing-based release retry is needed.
            self.assertEqual(released['listed'], 1)
            self.assertEqual((await driver.command('creation-retained-list'))['listed'], 1)
            acquired = await driver.command('creation-retained-acquire')
            self.assertEqual((acquired['result'], acquired['subject'], acquired['started'], acquired['closing']),
                             (0, subject, 0, 1))
            settled = await driver.command('creation-wait 5000')
            self.assertEqual((settled['result'], settled['phase'], settled['outcome'], settled['quiescent']),
                             (0, 3, 10, 1))
            self.assertEqual((await driver.command('creation-close'))['result'], 0)
        else:
            self.assertEqual(released['listed'], 0)
        self.assertEqual((await driver.command('creation-retained-list'))['listed'], 0)
        self.assertEqual((await driver.command('creation-retained-release'))['result'], 4)
        self.assertEqual(self.state.attempts, attempts)
        self.assertFalse(self.created_commits())
        await self.close(driver)
        self.assertEqual(self.state.attempts, attempts)
        self.assertFalse(self.created_commits())

    async def test_public_retained_creation_unknown_keeps_its_receipt_after_transfer(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        driver, subject, attempts = await self.retained_prepared()
        acquired = await driver.command('creation-retained-acquire')
        self.assertEqual((acquired['result'], acquired['subject']), (0, subject))
        self.assertEqual(self.state.attempts, attempts)
        self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await driver.command('creation-start'))['result'], 0)
        done = await driver.command('creation-wait 5000')
        self.assertEqual((done['result'], done['phase'], done['outcome'], done['state']), (0, 5, 19, 7))
        self.assertEqual(done['subject'], subject)
        self.assert_seed()
        self.assertEqual((await driver.command('creation-start'))['result'], 5)
        await self.close(driver)
        self.assertEqual(len(self.created_commits()), 1)
        self.assertEqual(self.state.attempts.get('CreateLayer'), attempts.get('CreateLayer', 0) + 1)
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), attempts.get('UpdateLayer', 0))

    async def test_public_creation_freezes_source_and_activates_once(self):
        driver, subject = await self.prepared()
        self.assertEqual((await driver.command('creation-start'))['result'], 0)
        done = await driver.command('creation-wait 5000')
        self.assertEqual((done['result'], done['phase'], done['outcome'], done['quiescent']), (0, 2, 0, 1))
        self.assertEqual(done['subject'], subject)
        self.assertEqual(done['fields'], {})  # Confirmation alone is not attachment readiness.
        self.assertEqual(done['state'], 1)
        self.assert_seed()
        self.assertEqual((await driver.command('creation-start'))['result'], 5)
        staged = await driver.command('creation-layer')
        self.assertEqual((staged['result'], staged['fields']), (0, {}))
        self.assertEqual(staged['state'], 1)
        activated = await driver.command('creation-activate')
        self.assertEqual((activated['result'], activated['fields'], activated['subject']), (0, {'counter': 17}, subject))
        attempts = dict(self.state.attempts)
        repeated = await driver.command('creation-activate')
        self.assertEqual((repeated['result'], repeated['fields']), (0, {'counter': 17}))
        self.assertEqual(self.state.attempts, attempts)
        await self.close(driver)
        self.assert_seed()

    async def test_actual_usd_create_new_finalizes_after_return_and_hands_its_target_to_the_layer(self):
        driver = await self.start()
        before = dict(self.state.attempts)
        created = await driver.command('creation-usd-new')
        self.assertEqual((created['hook_ok'], created['diagnosed'], created['immediate_state'],
                          created['held_finalizer'], created['dropped'], created['listed']), (1, 0, 1, 0, 0, 0), created)
        subject = created['subject']
        self.assertGreater(subject, 0)
        entry = self.assert_empty_created_seed()
        self.assertEqual(self.state.attempts.get('CreateLayer'), before.get('CreateLayer', 0) + 1)
        self.assertTrue(any(item['object'] == entry.object_id and item['cursor'] == 0 for item in self.state.reads))
        # The finalizer adopted the workflow into the returned target and freed its slot.
        self.assertEqual((await driver.command('creation-retained-list'))['listed'], 0)
        attempts = dict(self.state.attempts)
        saved = await driver.command('creation-sdf-save')
        self.assertEqual((saved['type'], saved['result'], saved['subject'], saved['state'], saved['fields']),
                         ('created-save', 0, subject, 2, {}))
        self.assertEqual(self.state.attempts.get('CreateLayer'), attempts.get('CreateLayer'))
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), before.get('UpdateLayer', 0))
        await self.close(driver)
        self.assert_empty_created_seed()

    async def test_actual_usd_create_new_target_is_read_only_until_activation(self):
        driver = await self.start()
        before = dict(self.state.attempts)
        created = await driver.command('creation-usd-new')
        # The driver attempts a plain Sdf edit of the unactivated target inside
        # the CreateNew host call; USD refuses it with a coding error.
        self.assertEqual((created['hook_ok'], created['read_only_refused'], created['immediate_state'],
                          created['state'], created['editable'], created['listed']), (1, 1, 1, 2, 1, 0), created)
        self.assert_empty_created_seed()
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), before.get('UpdateLayer', 0))
        edited = await driver.command('creation-edit probe 5')
        self.assertEqual((edited['editable'], edited['diagnosed'], edited['saved'], edited['fields']),
                         (1, 0, 0, {'probe': 5}), edited)
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), before.get('UpdateLayer', 0) + 1)
        commits = self.created_commits()
        self.assertEqual([entry.operation for entry in commits], ['CreateLayer', 'UpdateLayer'])
        self.assertEqual(decode_update(commits[-1].payload), {'probe': 5})
        await self.close(driver)

    async def test_actual_usd_create_new_lost_reply_retains_its_receipt_after_target_destruction(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        driver = await self.start()
        before = dict(self.state.attempts)
        self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        failed = await driver.command('creation-usd-new')
        self.assertEqual((failed['hook_ok'], failed['diagnosed'], failed['state'], failed['outcome'],
                          failed['pending'], failed['held_finalizer']), (0, 1, 7, 19, 1, 0), failed)
        subject = failed['subject']
        self.assert_empty_created_seed()
        attempts = dict(self.state.attempts)
        await driver.command('creation-retained-replace-context')
        acquired = await driver.command('creation-retained-acquire')
        self.assertEqual((acquired['phase'], acquired['outcome'], acquired['subject'], acquired['fields']),
                         (5, 19, subject, {}))
        self.assertEqual((await driver.command('creation-start'))['result'], 5)
        self.assertEqual(self.state.attempts, attempts)
        await self.close(driver)
        self.assertEqual(self.state.attempts.get('CreateLayer'), before.get('CreateLayer', 0) + 1)

    async def test_failed_create_new_keeps_the_canonical_key_until_released_then_reopens_without_create(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        driver = await self.start()
        self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        failed = await driver.command('creation-usd-new')
        self.assertEqual((failed['hook_ok'], failed['state'], failed['outcome']), (0, 7, 19), failed)
        attempts = dict(self.state.attempts)
        refused = await driver.command('creation-open')  # The retained workflow still holds the key.
        self.assertEqual((refused['opened'], refused['result']), (0, 5), refused)
        self.assertEqual(self.state.attempts, attempts)
        released = await driver.command('creation-retained-release')
        self.assertEqual((released['result'], released['listed']), (0, 0), released)
        reopened = await driver.command('creation-open')
        self.assertEqual((reopened['opened'], reopened['state'], reopened['fields']), (1, 2, {}), reopened)
        self.assertNotEqual(reopened['subject'], failed['subject'])
        self.assertEqual(self.state.attempts['CreateLayer'], attempts['CreateLayer'])  # Never recreated.
        self.assertEqual(self.state.attempts['SubscribeReadLayer'], attempts['SubscribeReadLayer'] + 1)
        await self.close(driver)

    async def test_actual_usd_create_new_dropped_target_fails_finalization_without_losing_receipt(self):
        await self.failed_new_finalization('creation-usd-new-drop', 10, returned=True)

    async def test_actual_usd_create_new_queue_then_throw_keeps_lease_until_late_job_retires(self):
        await self.failed_new_finalization('creation-usd-new-reject-dispatch', 13, returned=False)

    async def test_actual_usd_create_new_dropped_finalizer_keeps_the_failed_live_target(self):
        driver = await self.start()
        created = await driver.command('creation-usd-new-hold-finalizer')
        self.assertEqual((created['hook_ok'], created['diagnosed'], created['state'], created['held_finalizer'],
                          created['returned_layer_alive']), (1, 0, 1, 1, 1))
        subject = created['subject']
        self.assert_empty_created_seed()
        attempts = dict(self.state.attempts)
        dropped = await driver.command('creation-new-drop-finalizer')
        self.assertEqual((dropped['result'], dropped['subject'], dropped['listed']), (0, subject, 1))
        acquired = await driver.command('creation-retained-acquire')
        self.assertEqual((acquired['phase'], acquired['outcome'], acquired['state'], acquired['returned_layer_alive']),
                         (2, 0, 7, 1))
        # Explicit activation keeps the still-live failed target but must
        # preserve cancellation instead of publishing READY.
        activated = await driver.command('creation-activate')
        self.assertEqual((activated['result'], activated['state'], activated['returned_layer_alive']), (10, 7, 1))
        self.assertEqual((await driver.command('creation-activate'))['result'], 10)
        self.assertEqual(self.state.attempts, attempts)
        await self.close(driver)

    async def test_actual_usd_new_default_save_replaces_existing_and_keeps_its_receipt(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        original = update({'counter': 73})
        response = await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name='created.live')),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=original))]), metadata=WRITER, timeout=5)
        driver = await self.start()
        before = dict(self.state.attempts)
        self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        failed = await driver.command('creation-usd-new-save')
        self.assertEqual((failed['hook_ok'], failed['diagnosed'], failed['state'], failed['outcome'],
                          failed['returned_layer_alive'], failed['held_finalizer']), (0, 1, 7, 19, 1, 0), failed)
        subject = failed['subject']
        entry = self.assert_empty_created_seed(count=2)
        self.assertEqual(self.created_commits()[0].payload, original)
        self.assertNotEqual(entry.object_id, response.object_id)
        self.assertEqual(self.state.attempts.get('CreateLayer'), before.get('CreateLayer', 0) + 1,
                         'default pristine Save skipped creation of the replacement incarnation')
        attempts = dict(self.state.attempts)
        acquired = await driver.command('creation-retained-acquire')
        self.assertEqual((acquired['phase'], acquired['outcome'], acquired['subject'], acquired['returned_layer_alive']),
                         (5, 19, subject, 1))
        repeated = await driver.command('creation-new-repeat-save')
        self.assertEqual(repeated['hook_ok'], 0)
        self.assertEqual(self.state.attempts, attempts, 'repeated failed Save reissued Create')
        reset = await driver.command('creation-new-reset-target')
        self.assertEqual((reset['result'], reset['returned_layer_alive']), (0, 0))
        self.assertEqual((await driver.command('creation-start'))['result'], 5)
        self.assertEqual(self.state.attempts, attempts)
        await self.close(driver)
        self.assertEqual(self.state.attempts.get('CreateLayer'), before.get('CreateLayer', 0) + 1)

    async def failed_new_finalization(self, command, expected, *, returned):
        driver = await self.start()
        before = dict(self.state.attempts)
        failed = await driver.command(command)
        self.assertEqual(failed['hook_ok'], int(returned))
        self.assertEqual(failed['dropped'], int(returned))
        self.assertEqual((failed['state'], failed['outcome']), (7, expected), failed)
        subject = failed['subject']
        self.assert_empty_created_seed()
        attempts = dict(self.state.attempts)
        if returned:
            self.assertEqual((failed['immediate_state'], failed['held_finalizer']), (1, 0))
        else:
            self.assertEqual((failed['diagnosed'], failed['held_finalizer']), (1, 1))
            self.assertEqual((await driver.command('creation-retained-list'))['listed'], 1)
            late = await driver.command('creation-new-late-finalize')
            self.assertEqual((late['result'], late['subject'], late['listed']), (0, subject, 1))
        await driver.command('creation-retained-replace-context')
        acquired = await driver.command('creation-retained-acquire')
        self.assertEqual((acquired['phase'], acquired['outcome'], acquired['subject'], acquired['fields'],
                          acquired['state']), (2, 0, subject, {}, 7))
        self.assertEqual((await driver.command('creation-start'))['result'], 5)
        self.assertEqual(self.state.attempts, attempts, 'late finalization reissued service work')
        await self.close(driver)
        self.assertEqual(self.state.attempts.get('CreateLayer'), before.get('CreateLayer', 0) + 1)
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), before.get('UpdateLayer', 0))

    async def test_actual_usd_export_hands_off_its_destination_and_save_never_recreates(self):
        driver = await self.start()
        before = dict(self.state.attempts)
        exported = await driver.command('creation-usd-export 17')
        # The workflow is retained inside the outer call; its adoption job then
        # drops the destination, which no host reference holds, so it closes.
        self.assertEqual((exported['hook_ok'], exported['diagnosed'], exported['listed'], exported['outcome'],
                          exported['closed']), (1, 0, 0, 0, 1), exported)
        self.assertGreater(exported['subject'], 0)
        self.assert_seed()
        self.assertEqual(self.state.attempts.get('CreateLayer'), before.get('CreateLayer', 0) + 1)
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), before.get('UpdateLayer', 0))
        attempts = dict(self.state.attempts)
        # The source was mutated and destroyed after Export's bool return.
        opened = await driver.command('creation-open')
        self.assertEqual((opened['opened'], opened['result'], opened['state'], opened['fields']),
                         (1, 0, 2, {'counter': 17}), opened)
        # The open destination holds the canonical reservation: a second
        # Export is refused before retention or any service operation. Timing
        # Pings run on the endpoint's own schedule and are not operations.
        def operations():
            return {name: count for name, count in self.state.attempts.items() if name != 'Ping'}
        reopened = operations()
        duplicate = await driver.command('creation-usd-export 23')
        self.assertEqual((duplicate['hook_ok'], duplicate['diagnosed'], duplicate['listed'], duplicate['subject']),
                         (0, 1, 0, 0))
        self.assertEqual(operations(), reopened, 'duplicate Export issued another service operation')
        saved = await driver.command('creation-sdf-save')
        self.assertEqual((saved['result'], saved['fields']), (0, {'counter': 17}))
        self.assertEqual(self.state.attempts.get('CreateLayer'), attempts.get('CreateLayer'))
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), attempts.get('UpdateLayer', 0))
        await self.close(driver)
        self.assert_seed()

    async def test_actual_usd_export_adoption_job_dropped_by_the_host_reports_cancelled(self):
        driver = await self.start()
        dropped = await driver.command('creation-usd-export-drop-adoption 17')
        # The host released the queued adoption job without running it; the
        # job reports CANCELLED (10) and the workflow stays retained.
        self.assertEqual(dropped, {'type': 'usd-export-drop-adoption', 'hook_ok': 1, 'held': 1,
                                   'dropped_result': 10, 'listed': 1}, dropped)
        # The host takes the retained workflow back; closing the driver releases it.
        acquired = await driver.command('creation-retained-acquire')
        self.assertEqual(acquired['result'], 0, acquired)
        await self.close(driver)

    async def test_actual_usd_export_destination_found_in_its_host_turn_stays_open(self):
        driver = await self.start()
        before = dict(self.state.attempts)
        exported = await driver.command('creation-usd-export-find 17')
        # FindOrOpen in the Export's own host turn returns the destination the
        # workflow still holds; the adoption job then leaves it to that reference.
        self.assertEqual((exported['hook_ok'], exported['listed'], exported['closed'], exported['fields']),
                         (1, 0, 0, {'counter': 17}), exported)
        self.assert_seed()
        saved = await driver.command('creation-sdf-save')
        self.assertEqual((saved['result'], saved['subject'], saved['state'], saved['fields']),
                         (0, exported['subject'], 2, {'counter': 17}))
        self.assertEqual(self.state.attempts.get('CreateLayer'), before.get('CreateLayer', 0) + 1)
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), before.get('UpdateLayer', 0))
        await self.close(driver)

    async def test_actual_usd_export_reopen_after_its_host_turn_waits_for_the_closing_destination(self):
        driver = await self.start()
        before = dict(self.state.attempts)
        exported = await driver.command('creation-usd-export-reopen 17')
        # After the adoption job drops the destination, its closing actor keeps
        # the canonical reservation until it stops; a FindOrOpen in a later
        # host turn waits for that release and opens the exported content.
        self.assertEqual((exported['hook_ok'], exported['listed'], exported['first_opened'], exported['first_result'],
                          exported['closed'], exported['fields']), (1, 0, 1, 0, 0, {'counter': 17}), exported)
        self.assert_seed()
        self.assertEqual(self.state.attempts.get('CreateLayer'), before.get('CreateLayer', 0) + 1)
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), before.get('UpdateLayer', 0))
        await self.close(driver)

    async def test_reopen_is_refused_while_a_closing_layer_has_a_queued_host_publication(self):
        driver = await self.start()
        exported = await driver.command('creation-usd-export-find 17')
        self.assertEqual((exported['hook_ok'], exported['listed'], exported['fields']), (1, 0, {'counter': 17}), exported)
        self.assertEqual(await driver.command('creation-hold-host'), {'type': 'host-parked'})
        layer = self.state.sessions[self.session].layers['created.live']
        await self.stub.UpdateLayer(iter([
            pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name='created.live',
                                                             object_id=layer.object_id, client_id='peer')),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=update({'counter': 23})))]), metadata=WRITER, timeout=5)
        attempts = dict(self.state.attempts)
        # The peer update's publication waits on the parked host behind the
        # gate that drops the layer and reopens it. The closing actor keeps
        # its key until that host job retires, so the reopen in the gate is
        # refused with ALREADY_EXISTS instead of waiting on its own host; after
        # the old layer reports CLOSED a reopen succeeds.
        reopened = await driver.command('creation-reopen-parked')
        self.assertEqual((reopened['opened'], reopened['result'], reopened['reopened'], reopened['state'],
                          reopened['fields']), (0, 5, 1, 2, {'counter': 23}), reopened)
        self.assertEqual(self.state.attempts.get('CreateLayer'), attempts.get('CreateLayer'))
        await self.close(driver)

    async def test_actual_usd_export_rejects_comment_and_arguments_before_retention_or_rpc(self):
        driver = await self.start()
        attempts = dict(self.state.attempts)
        for command in ('creation-usd-export-comment 17', 'creation-usd-export-args 17'):
            rejected = await driver.command(command)
            self.assertEqual((rejected['hook_ok'], rejected['listed'], rejected['subject']), (0, 0, 0), rejected)
            self.assertEqual(self.state.attempts, attempts)
            self.assertFalse(self.created_commits())
        # Rejected options must not consume the canonical reservation.
        exported = await driver.command('creation-usd-export 17')
        self.assertEqual((exported['hook_ok'], exported['listed']), (1, 0))
        self.assertEqual((await driver.command('creation-open'))['fields'], {'counter': 17})
        await self.close(driver)
        self.assert_seed()

    async def test_actual_usd_export_lost_committed_reply_retains_the_original_seed_receipt(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        driver = await self.start()
        before = dict(self.state.attempts)
        self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        exported = await driver.command('creation-usd-export 17')
        self.assertEqual((exported['hook_ok'], exported['diagnosed'], exported['listed'], exported['outcome'],
                          exported['state'], exported['pending']), (0, 1, 1, 19, 7, 1), exported)
        subject = exported['subject']
        self.assertGreater(subject, 0)
        self.assertGreater(exported['incident'], 0)
        self.assert_seed()
        attempts = dict(self.state.attempts)
        duplicate = await driver.command('creation-usd-export 23')
        self.assertEqual((duplicate['hook_ok'], duplicate['subject'], duplicate['outcome']), (0, subject, 19))
        self.assertEqual(self.state.attempts, attempts, 'failed Export implicitly retried the mutation')
        await driver.command('creation-retained-replace-context')
        acquired = await driver.command('creation-retained-acquire')
        self.assertEqual((acquired['result'], acquired['phase'], acquired['outcome'], acquired['subject']),
                         (0, 5, 19, subject))
        self.assertEqual(acquired['fields'], {})
        self.assertEqual((await driver.command('creation-start'))['result'], 5)
        self.assertEqual(self.state.attempts, attempts)
        await self.close(driver)
        self.assertEqual(len(self.created_commits()), 1)
        self.assertEqual(self.state.attempts.get('CreateLayer'), before.get('CreateLayer', 0) + 1)

    async def test_actual_usd_export_replaces_external_incarnation_and_old_receipt_fails_attachment(self):
        original, subject = await self.prepared()
        self.assertEqual((await original.command('creation-start'))['result'], 0)
        confirmed = await original.command('creation-wait 5000')
        self.assertEqual((confirmed['phase'], confirmed['outcome'], confirmed['fields']), (2, 0, {}))
        self.assert_seed()
        old_commit = self.created_commits()[0]
        replacement = await self.start()  # Separate process, canonical reservation cannot conflict locally.
        exported = await replacement.command('creation-usd-export 23')
        self.assertEqual((exported['hook_ok'], exported['listed'], exported['outcome']), (1, 0, 0))
        reopened = await replacement.command('creation-open')
        self.assertEqual(reopened['fields'], {'counter': 23})
        entries = self.created_commits()
        self.assertEqual(len(entries), 2)
        self.assertEqual([entry.operation for entry in entries], ['CreateLayer', 'CreateLayer'])
        self.assertEqual([entry.sequence for entry in entries], [1, 1])
        self.assertNotEqual(entries[1].object_id, old_commit.object_id)
        for entry, counter in zip(entries, (17, 23)):
            seed = graph_decode(entry.payload)
            self.assertEqual((seed.is_diff, seed.base_version, seed.source_format), (True, 0, 'usda'))
            self.assertEqual([(field.section, field.key, field.value) for field in seed.fields], [(1, 'counter', counter)])
        attempts = dict(self.state.attempts)
        # Opening a known-old receipt after replacement receives a new-object
        # header. An already-running reference stream uses a different policy.
        refused = await original.command('creation-layer')
        self.assertEqual((refused['result'], refused['subject'], refused['phase'], refused['outcome']),
                         (16, subject, 2, 0))
        self.assertEqual(refused['fields'], {})
        self.assertEqual(self.state.attempts.get('CreateLayer'), attempts.get('CreateLayer'))
        self.assertEqual((await replacement.command('creation-sdf-save'))['fields'], {'counter': 23})
        await self.close(original)
        await self.close(replacement)
        self.assertEqual(self.created_commits(), entries)

    async def test_public_creation_unknown_never_retries(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        driver, subject = await self.prepared()
        self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await driver.command('creation-start'))['result'], 0)
        done = await driver.command('creation-wait 5000')
        self.assertEqual((done['result'], done['phase'], done['outcome'], done['quiescent']), (0, 5, 19, 1))
        self.assertEqual(done['subject'], subject)
        self.assertGreater(done['incident'], 0)
        self.assertEqual(done['pending'], 1)
        self.assert_seed()
        attempts = dict(self.state.attempts)
        self.assertEqual((await driver.command('creation-start'))['result'], 5)
        self.assertNotEqual((await driver.command('creation-layer'))['result'], 0)
        self.assertEqual(self.state.attempts, attempts)
        await self.close(driver)
        self.assertEqual(len(self.created_commits()), 1)
        self.assertGreater(done['request'], 0)
        self.assertGreater(done['attempt'], 0)

    async def test_public_creation_timeout_preserves_original_attempt(self):
        driver, _ = await self.prepared()
        fault = self.state.plan('CreateLayer', 'before', pause=True)
        attempts = self.state.attempts.get('CreateLayer', 0)
        self.assertEqual((await driver.command('creation-start'))['result'], 0)
        await asyncio.wait_for(fault.reached.wait(), 5)
        timed = await driver.command('creation-wait 10')
        self.assertEqual((timed['result'], timed['phase'], timed['quiescent']), (9, 1, 0))
        self.assertFalse(self.created_commits())
        self.assertEqual(self.state.attempts['CreateLayer'], attempts + 1)
        fault.release.set()
        done = await driver.command('creation-wait 5000')
        self.assertEqual((done['result'], done['phase']), (0, 2))
        self.assert_seed()
        self.assertEqual(self.state.attempts['CreateLayer'], attempts + 1)
        await self.close(driver)

    async def test_public_prepared_close_never_issues_creation(self):
        driver, _ = await self.prepared()
        attempts = self.state.attempts.get('CreateLayer', 0)
        closed = await driver.command('creation-close')
        self.assertIn(closed['result'], (0, 14))
        self.assertNotEqual((await driver.command('creation-start'))['result'], 0)
        await self.close(driver)
        self.assertFalse(self.created_commits())
        self.assertEqual(self.state.attempts.get('CreateLayer', 0), attempts)

    async def test_public_creation_from_live_layer_preserves_empty_source_format(self):
        await self.stub.UpdateLayer(iter([
            pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name='root.live',
                                                             object_id=1, client_id='peer')),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=update({'counter': 31})))]), metadata=WRITER, timeout=5)
        driver, _ = await self.prepared('creation-prepare-live')
        # The seed freezes at prepare; later live-layer updates must not become
        # the seed, and its empty format must not become live_owned_scene.
        await self.stub.UpdateLayer(iter([
            pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name='root.live',
                                                             object_id=1, client_id='peer')),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=update({'counter': 72})))]), metadata=WRITER, timeout=5)
        advanced = await driver.command('wait 1 counter 72')
        self.assertEqual(advanced['fields'], {'counter': 72})
        self.assertEqual((await driver.command('creation-start'))['result'], 0)
        self.assertEqual((await driver.command('creation-wait 5000'))['phase'], 2)
        self.assert_seed(31, '')
        self.assertEqual((await driver.command('creation-layer'))['result'], 0)
        activated = await driver.command('creation-activate')
        self.assertEqual((activated['result'], activated['fields']), (0, {'counter': 31}))
        await self.close(driver)
