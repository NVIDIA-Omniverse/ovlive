# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh public-header-only consumers of the packaged production shared plugin."""
import asyncio
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
import unittest
from urllib.parse import quote

import grpc
from service import Chained, Commit, State, Principal, ReferenceServer, pb, rpc
from scene_wire import Model, decode_update, empty_snapshot, update
from graph_service import GraphService, GraphState, GraphServer
from scene_graph_wire import decode as graph_decode
import scene_graph_model as graph_model
import test_scene as scene_support
import tls_fixture
from writer_history import (generate as writer_plan, account as account_writers,
                            load as load_writer_plan, validate as validate_writer_plan, WriterMismatch)
from writer_reduction import reduce_history as reduce_writers, publish as publish_writer, period_domain, round_domain
import period_plans
from period_plans import generate as rotation_writer_plan
import plan_replay
import round_plans
import rotation_history
from tools.live import writer_campaign as campaign

WRITER = (("authorization", "Bearer synthetic-writer"),)


class CorruptWriterService(scene_support.CompactSceneService):
    """Named test-only service defect: corrupt one edit before committing it, committing -1 for `field`, which only
    that edit authors, in whichever update carries it.

    Real plugin processes still admit, issue and acknowledge the original edit.
    This exercises semantic failure reduction; no production fault switch exists.
    """
    field = 'owned_0_2'

    async def chunks(self, stream, context):
        header, payload = await super().chunks(stream, context)
        # An update's header is plain, or Chained when the endpoint chains its writes.
        if not isinstance(header, (pb.UpdateLayerHeader, Chained)):
            return header, payload
        fields = decode_update(payload)
        if self.corrupts(fields):
            fields[self.field] = -1
            payload = update(fields)
        return header, payload

    def corrupts(self, fields):
        return self.field in fields


class CorruptRotationWriterService(CorruptWriterService):
    """The queued-rotation defect `corrupt-owned-w0:6`: writer 0's round-6 edit, an owned field, commits -1. Its
    field is written once, so no later write of a coalesced run folds it away, whatever the run's shape."""
    field = 'owned_0_6'


class CorruptPeriodWriterService(CorruptWriterService):
    """The period-plan defect `corrupt-w0:4` of the stream-age and long-rotations cases: writer 0's ordinal-4 edit
    commits -1, on whichever backend its period selects. A period plan writes each owned field every eighth period, so
    only the update carrying `state.corrupt_item`, that edit's planned field and value, is corrupted; a plan without
    the edit sets it to None."""
    @property
    def field(self):
        return self.state.corrupt_item[0]

    def corrupts(self, fields):
        item = getattr(self.state, 'corrupt_item', None)
        return item is not None and fields.get(item[0]) == item[1]


# Named test-only defects of uncertain-write plans (rotation_history.FAULTS), as CutWriterService modes.
CUT_DEFECTS = {'apply-cut-w0:0': 'apply', 'lose-cut-w0:0': 'lose', 'leak-cut-w0:0': 'leak'}


class CutWriterService(scene_support.CompactSceneService):
    """Named test-only service defects at an uncertain-write cut, on the update whose fields are
    `state.cut_defect[1]`.

    `apply` buffers each upload before its handler begins and commits that update when the cut cancels its handler
    held before execution; `lose` passes that update through its commit step without committing it; `leak` commits
    it and records its attempt as `state.cut_attempt`, whose fault the harness releases before the cut. Real plugin
    processes still issue the original edit. This exercises cut-oracle failure reduction; no production fault switch
    exists.
    """
    def defective(self, payload, mode):
        defect = getattr(self.state, 'cut_defect', None)
        return defect is not None and defect[0] == mode and decode_update(payload) == defect[1]

    async def UpdateLayer(self, stream, context):
        if getattr(self.state, 'cut_defect', (None,))[0] != 'apply':
            return await super().UpdateLayer(stream, context)
        messages = [message async for message in stream]

        async def replay():
            for message in messages:
                yield message
        try:
            return await super().UpdateLayer(replay(), context)
        except asyncio.CancelledError:
            header = next(m.header for m in messages if m.WhichOneof('message_type') == 'header')
            payload = b''.join(m.chunk.data for m in messages if m.WhichOneof('message_type') == 'chunk')
            if self.defective(payload, 'apply'):
                # State.commit, without the lock a cancelled handler cannot take; wake() notifies the waiters.
                state, layer = self.state, self.state.sessions[header.session_id].layers[header.layer_name]
                entry = Commit(len(state.ledger) + 1, 'UpdateLayer', header.session_id, timestamp_us=state.clock(),
                               layer=header.layer_name, object_id=layer.object_id,
                               sequence=layer.commits[-1].sequence + 1, payload=payload, writer=header.client_id,
                               client_timestamp_us=header.timestamp_microseconds_since_epoch)
                state.ledger.append(entry)
                state.retained_bytes += len(payload)
                layer.commits.append(entry)
                state.wake()
            raise

    async def apply_update(self, method, attempt, session, layer, header, payload, context):
        if self.defective(payload, 'lose'):
            return pb.UpdateLayerResponse(sequence_num=layer.commits[-1].sequence + 1,
                                          server_timestamp_microseconds_since_epoch=self.state.clock())
        if self.defective(payload, 'leak'):
            self.state.cut_attempt = attempt
        return await super().apply_update(method, attempt, session, layer, header, payload, context)


async def fresh_trial(fixture, run):
    """One reduction trial: set up the fresh `fixture`, await `run`, a coroutine of its, and run every cleanup the
    fixture registered, even after a partial setup; returns the reducer's result.

    A mismatch counts only once every native writer has closed through the checked path (PluginTests.retire_writers);
    a writer that cannot close or exited abnormally leaves the trial unresolved, with the mismatch's signature as
    `mismatch`.
    """
    result = dict(outcome='unresolved', cleanup_complete=False)
    try:
        await fixture.asyncSetUp()
        await run
        result.update(outcome='passed')
    except WriterMismatch as error:
        try:
            await fixture.retire_writers()
            # The reducer compares only the signature; the evidence stays in the trial's result.json.
            result.update(outcome='mismatch', signature=error.signature, **({'evidence': error.evidence}
                                                                            if error.evidence else {}))
        except Exception as failure:
            result.update(error=repr(failure), mismatch=error.signature)
    except Exception as error:
        result.update(error=repr(error))
    finally:
        run.close()  # A setup failure leaves the coroutine unawaited.
        try:
            # Run all registered cleanups, even on partial setup failure.
            while fixture._cleanups:
                callback, args, kwargs = fixture._cleanups.pop()
                await callback(*args, **kwargs)
            result['cleanup_complete'] = True
        except Exception as error:
            result.update(outcome='unresolved', cleanup_complete=False, cleanup_error=repr(error))
    return result


async def export_layer(driver, path):
    """`driver`'s reply to an export of its layer to `path`. SdfLayer::Export resolves the path through TfRealPath,
    which on Windows fails for a path of MAX_PATH (260) characters or more, and a reducer's standalone replay nests its
    artifacts deeper than that; so on Windows the writer exports into a short temporary directory, which the suite's
    scratch directory holds until the suite ends, and the file is copied to `path`. The writer re-reads its export when
    it closes, so the exported file stays where it wrote it."""
    if os.name != 'nt':
        return await driver.command(f'export {path}')
    short = Path(tempfile.mkdtemp()) / 'export.usda'
    exported = await driver.command(f'export {short}')
    if exported['result'] == 0:
        shutil.copyfile(short, path)
    return exported


async def crash_beside_mismatch(fixture, run):
    """A test-only evaluator run: await `run()`, a coroutine of the fixture's, and on its WriterMismatch kill the
    fixture's first running native writer before the mismatch propagates, as a writer that crashed after its last
    reply beside a genuine mismatch."""
    try:
        await run()
    except WriterMismatch:
        driver = next(driver for driver in fixture.drivers if driver.process.returncode is None)
        await scene_support.kill_writer(driver.process)
        raise


# The plugin test of each root-field rotation case that reduces its failures.
ROTATION_TESTS = {
    'queued-rotations': 'test_seeded_writers_resume_queued_edits_through_two_backend_rotations',
    'queued-rotations-coalesced': 'test_seeded_coalesced_writers_resume_queued_edits_through_two_backend_rotations',
    'uncertain-before': 'test_rotation_before_execution_retains_uncertain_and_queued_writer_intents',
    'uncertain-after': 'test_rotation_after_commit_retains_uncertain_and_queued_writer_intents',
}
# The plugin test of each period-plan case that reduces its failures.
PERIOD_TESTS = {
    'long-rotations': 'test_seeded_writers_converge_through_many_backend_rotations',
    'stream-age': 'test_seeded_writers_and_joins_survive_stream_age_expiries_after_server_token_expiry',
}


async def writer_tasks(*operations):
    """Own every trial command until success or sibling cancellation settles."""
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(operation) for operation in operations]
    return [task.result() for task in tasks]


async def client_update_attempts(driver):
    """UpdateLayer attempts one client process issued, from its public diagnostics."""
    metrics = (await driver.command('diagnostics'))['snapshot']['metrics']
    return sum(item['count'] for item in metrics['rpc_attempts_by_operation'] if item['operation'] == 4)


async def issued_at_least(driver, least):
    """UpdateLayer attempts once a client has issued at least `least`: its worker issues each admitted write after
    the write's local publication, so the count is polled under a failure-only watchdog."""
    deadline = asyncio.get_running_loop().time() + 5
    while True:
        attempts = await client_update_attempts(driver)
        if attempts >= least or asyncio.get_running_loop().time() > deadline:
            return attempts


# The chain window a layer's first write issues under (design section 13). Admitted writes never exceed it while no
# acknowledgment widens it, and at most one waits to be issued, so a writer whose first write is held and that
# authors more than one block per window slot must coalesce.
FIRST_CHAIN_WINDOW = 8


def chained_commits(state, client_id):
    """The service's commits of `client_id`'s writes that named a predecessor, from its chain log."""
    return [event for event in state.chain_log
            if event['event'] == 'commit' and event['client_id'] == client_id and event['previous_operation_id']]


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


async def settled_snapshot(driver):
    """A diagnostics snapshot taken once the runtime is settled: both executors have no queued or running job and
    no application is admitted, in this snapshot and in the one before it.

    A job can still be accounted as running just after its callback answered the driver. A host publication retires
    its application on the host thread, and the scheduler thread queues the actor that settles the retirement, so a
    snapshot can find both executors idle while an application whose fields the driver already read is still
    admitted with its pending bytes reserved. A snapshot copies the metrics before the executor counters, so a
    settled snapshot's metrics can predate the end of the job that settled the last application; the next snapshot
    follows that job. A job or application that never finishes fails the watchdog.
    """
    deadline = time.monotonic() + 10  # Failure-only watchdog.
    previous = False
    while True:
        snapshot = (await driver.command('diagnostics'))['snapshot']
        executors, applications = snapshot['metrics']['executors'], snapshot['metrics']['applications']
        settled = (all(executor['queued'] == 0 and executor['running'] == 0 for executor in executors.values()) and
                   all(application['current'] == 0 for application in applications))
        if settled and previous:
            return snapshot
        previous = settled
        if time.monotonic() >= deadline:
            raise AssertionError(f'runtime stayed busy: executors {executors}, applications {applications}')


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


class RotationRun:
    """Writers rotating between two backends behind one RotatingProxy.

    Each rotation parks new connections, so every writer's read resumes once,
    queues one planned edit without issuing it, recovers on the other backend
    and converges before the rotation ends.
    """
    def __init__(self, test, plan, proxy, backends, history):
        self.test, self.plan, self.proxy, self.backends, self.history = test, plan, proxy, backends, history
        self.clients = plan['client_count']
        self.drivers, self.status, self.generations = [], [], []
        self.seen, self.initial_ms, self.maximum_ms = [0] * self.clients, [0] * self.clients, [0] * self.clients
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
        self.maximum_ms[index] = configuration['reconnect_maximum_ms']
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

    async def rotate(self, rotation, *, period=None, gated, relaxed=False, client_updates=None, destination=None):
        """One outage and recovery with exact accounting of the writers' planned edits.

        rotation counts the rotations run, this one included, and runs each
        writer's plan edit at that position; period is the seeded period it
        runs, rotation by default, whose parity names the destination backend
        (period_plans) and which the accounting and resume signatures name.
        gated holds every caught-up read reply until all writers reconnected,
        with no write issued. relaxed admits the extra retries of a reconnect
        whose credentials the service rejected: at least one retry_scheduled
        and a changed incident. client_updates are the UPDATE attempts each
        writer has issued after this rotation, the rotation number by default.
        destination, a test-only defect's, names the backend the proxy forwards
        to in place of the period's.
        """
        test, state, clients, drivers = self.test, self.test.state, self.clients, self.drivers
        period = rotation if period is None else period
        name = 'B' if period % 2 else 'A'
        before = len(test.commits())
        read_begin = len(state.scene_reads)
        read_attempt = state.attempts.get('SubscribeReadLayer', 0)
        write_attempt = state.attempts.get('UpdateLayer', 0)
        read_faults = [state.plan('SubscribeReadLayer', 'before_reply', attempt=read_attempt + i + 1,
                                  pause=True) for i in range(clients)] if gated else []
        try:
            retired = await asyncio.wait_for(self.proxy.rotate_and_wait(self.backends[destination or name],
                                                                        pause=True), 5)
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
        # The rotation's own check, once every writer is ready on the destination, before the plain read checks:
        # each writer resumed reading there at the frontier before the rotation.
        reads = state.scene_reads[read_begin:]
        period_plans.check_rotation_resume(period, reads, clients, name, before)
        test.assertEqual(len(reads), clients)
        test.assertEqual(len({read['writer'] for read in reads}), clients)
        test.assertTrue(all(read['object'] == 1 for read in reads), reads)
        saves = await writer_tasks(*(driver.command('save 5000') for driver in drivers))
        test.assertTrue(all(result['result'] == 0 for result in saves), saves)
        commits = test.commits()[before:]
        decoded = [{'sequence': commit.sequence, 'writer': commit.writer, 'fields': decode_update(commit.payload)}
                   for commit in commits]
        partial = dict(self.plan, writers=[[op] for op in ops])
        # The accounting comes before the plain commit count, so a lost or repeated commit raises its
        # WriterMismatch, which names the period.
        with period_plans.in_period(period):
            changed, _ = account_writers(partial, [{'edit': op['edit'], 'ordinal': 1} for op in ops], decoded,
                                        base_sequence=before)
        test.assertEqual(len(commits), clients)
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
            if relaxed:
                retry = [r['fields'] for r in reads if r['event_name'] == 'retry_scheduled']
                resumed = [r['fields'] for r in reads if r['event_name'] == 'stream_resumed']
                test.assertEqual(len(resumed), 1, reads)
                test.assertGreaterEqual(len(retry), 1, reads)
            else:
                # The first retry is drawn from the first backoff window, so
                # the attempt count reset at the last recovery; a retry that
                # fails before reaching a backend is the next attempt.
                retry, resumed = period_plans.check_rotation_reconnects(
                    reads, self.generations[index], self.initial_ms[index], self.maximum_ms[index])
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
        self.server = (GraphServer(self.state, "A",
                                   service_factory=getattr(self, 'graph_service_factory', GraphService))
                       if getattr(self, 'graph_creation', False) else
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

    async def start_answered(self):
        """Starts a driver and returns once the service has counted the Ping in which its endpoint learns the service's
        capabilities and reached that Ping's before_reply boundary. The driver reports ready before its layer actor
        sends that Ping, so otherwise the Ping may reach the service during whatever the test does next."""
        capabilities = self.state.plan("Ping", "before_reply")
        driver = await PluginTests.start(self)
        await asyncio.wait_for(capabilities.reached.wait(), 5)
        return driver

    async def wait_fields(self, driver, expected):
        command = "wait " + str(len(expected)) + "".join(f" {key} {value}" for key, value in sorted(expected.items()))
        result = await driver.command(command)
        self.assertEqual(result["fields"], expected)
        self.trace["states"].append(result)
        return result

    def release_holds(self):
        """Release every planned service fault and open every rotating proxy's gate, so no writer's close waits on a
        hold the fixture still keeps."""
        for fault in self.state.faults:
            fault.release.set()
        for proxy in self.proxies:
            if hasattr(proxy, 'gate'):
                proxy.gate.set()

    async def retire_writers(self):
        """Release the fixture's holds, then close every native writer still running through the checked path (close)
        and check that every writer that already exited did so cleanly; raises when any writer failed, so no
        reduction accepts a mismatch beside a crashed writer.

        The mismatch is final before this runs, so releasing the holds changes no signature; a released write
        commits or is cancelled by its writer's close, which cancels every call still in flight."""
        self.release_holds()
        for driver in self.drivers:
            if driver.process.returncode is None:
                await self.close(driver)
            else:
                await driver.drainer
                self.assertIsNone(driver.failure)
                self.assertEqual(driver.process.returncode, 0, driver.log.read_text()[-8192:])

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
        # The driver saves off the host, which returns BUSY while a host edit is unflushed; the second block's flush
        # coalesces while the first block's write fills the issuance window, so the save waits for its admission.
        self.assertEqual((await second.command("save-admitted 5000"))["result"], 0)
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
        first, second = await self.start_answered(), await self.start_answered()
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
        await self.public_writers(coalesced=False)

    async def test_seeded_coalesced_public_writers_account_every_edit_and_conflict(self):
        """The campaign with chained writes advertised and plain USD authoring, so each writer chains and coalesces."""
        await self.public_writers(coalesced=True)

    async def public_writers(self, coalesced):
        replay = os.environ.get('LIVE_TEST_WRITER_PLAN')
        generated = writer_plan(self.seed, clients=getattr(self, 'writer_clients', 8), steps=getattr(self, 'writer_steps', 16))
        if coalesced:
            generated['coalesced_writes'] = True
        plan = load_writer_plan(Path(replay)) if replay else self.writer_inputs(generated)
        # Explicit replay uses a fresh fixture too, so a named test-only service
        # defect can be reproduced without changing the shared default service.
        if replay:
            result = await self.writer_trial(plan, self.artifact / 'writer-replay')
            publish_writer(self.artifact / 'writer-replay-result.json', result)
            if result['outcome'] == 'mismatch':
                raise WriterMismatch(evidence=result.get('evidence'), **result['signature'])
            self.assertEqual(result['outcome'], 'passed', result)
            return
        await self.run_writers_with_reduction(plan)

    async def run_writers_with_reduction(self, plan):
        try:
            await self.exercise_writers(plan)
        except WriterMismatch as error:
            publish_writer(self.artifact / 'writer-mismatch.json', dict(signature=error.signature,
                                                                       evidence=error.evidence))
            # The mismatch counts only once every writer closed cleanly; a crashed writer raises a plain error here,
            # which is not reduced. The mismatch is published first, so it stays on record either way.
            await self.retire_writers()
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

    async def writer_trial(self, plan, directory, *, automatic_reduction=False, crash=False):
        """One fresh fixture, service and set of native writers executing `plan`; the reducer's evaluator. `crash`, a
        test's, kills a writer beside a mismatch (crash_beside_mismatch)."""
        validate_writer_plan(plan)
        directory.mkdir(parents=True, exist_ok=True)
        fixture = PluginTests('test_seeded_public_writers_account_every_edit_and_conflict')
        fixture.artifact_override = directory
        fixture.seed_override = plan['master_seed']
        if plan.get('test_service_fault') == 'corrupt-owned-0-2':
            fixture.writer_service_factory = CorruptWriterService
        if crash:
            return await fresh_trial(fixture, crash_beside_mismatch(fixture, functools.partial(
                fixture.exercise_writers, plan)))
        return await fresh_trial(fixture, fixture.run_writers_with_reduction(plan) if automatic_reduction else
                                 fixture.exercise_writers(plan))

    async def exercise_writers(self, plan):
        # Without coalesced_writes the campaign binds one intent per command and holds each writer's first RPC: one
        # unchained write at a time. With it the endpoint advertises chained writes and each command is a plain USD
        # change block, so a writer chains its writes behind the held first one and coalesces blocks while its window
        # is full; each commit is then a contiguous run of the writer's edits.
        coalesced = plan.get('coalesced_writes', False)
        self.state.advertise_capabilities = coalesced
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
            drivers, wire = [], []
            for _ in plan['writers']:
                reads = len(self.state.scene_reads)
                drivers.append(await self.start())
                # A writer reads and writes under one client ID; its first read names it.
                ids = {read['writer'] for read in self.state.scene_reads[reads:]}
                self.assertEqual(len(ids), 1, self.state.scene_reads[reads:])
                wire.append(ids.pop())
            for index, driver in enumerate(drivers):
                record('writer-ready', writer=index, pid=driver.process.pid, client_id=wire[index])
            if coalesced:
                # Each writer's first write is held at its commit step; its chained successors wait behind it.
                faults = [self.state.plan_write('hold', position=1, writer=client_id) for client_id in wire]
            else:
                faults = [self.state.plan('UpdateLayer', 'before', attempt=i + 1, pause=True)
                          for i in range(len(drivers))]

            admitted = [0] * len(drivers)

            async def author(index, driver):
                for op in plan['writers'][index]:
                    record('submit', **op)
                    # One plain USD change block; its flush job has run when the
                    # command returns. Bound to its own intent, the block is one
                    # admitted intent, and with every first RPC held no admitted
                    # intent retires; coalesced, it may join a later intent.
                    command = 'usd-set' if coalesced else 'usd-set-admit'
                    result = await driver.command(f"{command} 1 {op['field']} {op['value']}")
                    self.assertEqual((result['result'], result['status_result']), (0, 0), result)
                    admitted[index] += 1
                    if not coalesced:
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
                if not coalesced:
                    self.assertEqual(self.state.attempts['UpdateLayer'], len(drivers))
                self.assertEqual(len(self.commits()), 1, 'held requests committed before release')
                # All independent processes have admitted a complete queue while
                # every writer's first issued RPC is simultaneously held.
                for i, driver in enumerate(drivers):
                    state = await driver.command('state')
                    if coalesced:
                        # Once the window is full, later blocks coalesce instead of admitting.
                        self.assertTrue(1 <= state['pending'] <= len(plan['writers'][i]), state)
                        if len(plan['writers'][i]) > FIRST_CHAIN_WINDOW + 1:
                            self.assertLess(state['pending'], len(plan['writers'][i]), state)
                    else:
                        self.assertEqual(state['pending'], len(plan['writers'][i]), state)
                    own = {op['field']: op['value'] for op in plan['writers'][i]}
                    # The host content holds every edit this writer authored.
                    self.assertEqual(state['fields'], own)
                    # Each writer's first update is issued and held; unchained, the rest wait.
                    if coalesced:
                        # A writer of two or more blocks issues a second write, chained behind the held one.
                        attempts = await issued_at_least(driver, min(2, len(plan['writers'][i])))
                        self.assertTrue(min(2, len(plan['writers'][i])) <= attempts <= len(plan['writers'][i]),
                                        attempts)
                    else:
                        attempts = await client_update_attempts(driver)
                        self.assertEqual(attempts, 1)
                    record('held-state', writer=i, state=state, attempts=attempts)
                late = await self.start()
                record('late-observer-ready', pid=late.process.pid)
            finally:
                for index in plan['release_order']:
                    record('release-first-rpc', attempt=index + 1)
                    faults[index].release.set()

            # Observe service progress explicitly before checking local save
            # retirement. A correctness campaign must not require hundreds of
            # instrumented, contending writes to finish in a save's five-second
            # deadline. The condition is notified by actual commits; timeout is
            # only a failure watchdog, never evidence of successful progress.
            # Unchained, every edit is one commit, so the boundary is known in
            # advance; coalesced, it is the commit that carries every writer's
            # last edit, after which no writer has anything left to write.
            lasts = [(writer[-1]['field'], writer[-1]['value']) for writer in plan['writers']]
            boundary = 1 + sum(map(len, plan['writers']))

            def committed():
                if not coalesced:
                    return len(self.commits()) >= boundary
                carried = set()
                for commit in self.commits()[1:]:
                    carried.update(decode_update(commit.payload).items())
                return all(last in carried for last in lasts)
            # Workload-sized: 32 writers' 512 commits took 28 to 30 seconds on Windows under asyncio debug mode
            # (4.5 on Linux), so the watchdog allows a quarter second per edit and never less than 30 seconds.
            async with asyncio.timeout(max(30, sum(map(len, plan['writers'])) / 4)):
                async with self.state.changed:
                    await self.state.changed.wait_for(committed)
            if coalesced:
                boundary = len(self.commits())
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
            # A mismatch counts only once every native writer closes cleanly (retire_writers), so a crash cannot
            # masquerade as loss.
            expected, lineage = account_writers(plan, admissions, commits, coalesced=coalesced)
            self.assertEqual(len(self.commits()), boundary)
            self.assertEqual(self.state.attempts['UpdateLayer'], boundary - 1)
            self.assertEqual(expected, self.oracle())
            record('edit-accounting', lineage=lineage, expected=expected)
            for i, driver in enumerate(drivers + [late]):
                state = await self.wait_fields(driver, expected)
                self.assertEqual((state['result'], state['pending']), (0, 0), state)
                record('converged', writer=i, state=state)
            # Unchained, each change block issued exactly one UpdateLayer;
            # coalesced, each commit of a writer is one of its UpdateLayer
            # calls. The late observer issued none.
            for i, driver in enumerate(drivers):
                own = sum(commit['writer'] == wire[i] for commit in commits)
                attempts = await client_update_attempts(driver)
                if attempts != (own if coalesced else len(plan['writers'][i])):
                    # Each UpdateLayer outcome by result, so a mismatch names the attempt that did not commit.
                    metrics = (await driver.command('diagnostics'))['snapshot']['metrics']
                    record('attempt-mismatch', writer=i, attempts=attempts, commits=own,
                           outcomes=[item for item in metrics['rpc_by_operation_result'] if item['operation'] == 4])
                self.assertEqual(attempts, own if coalesced else len(plan['writers'][i]), (i, own))
                self.assertTrue(1 <= own <= len(plan['writers'][i]), (i, own))
                if not coalesced:
                    continue
                # The mode took effect: writes behind the held first one were chained, and a writer of more blocks
                # than the first window holds coalesced some of them.
                edits = len(plan['writers'][i])
                writes = (await driver.command('diagnostics'))['snapshot']['metrics']['layer_writes']
                record('layer-writes', writer=i, metrics=writes, chained=len(chained_commits(self.state, wire[i])))
                if edits >= 2:
                    self.assertTrue(chained_commits(self.state, wire[i]), (i, self.state.chain_log))
                    self.assertGreater(writes['window_high_water'], 1, writes)
                if edits > FIRST_CHAIN_WINDOW + 1:
                    self.assertLess(own, edits, (i, own))
                    self.assertGreater(writes['coalesced_turns'], 0, writes)
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
        def compared(result):
            # A trial's evidence depends on the plan and the commit order, so a replay matches the rest.
            return {key: value for key, value in result.items() if key != 'evidence'}
        first = await self.writer_trial(plan, directory, automatic_reduction=True)
        self.assertEqual(compared(first), dict(outcome='mismatch', cleanup_complete=True,
                                               signature=dict(reason='unexplained committed field', field='owned_0_2')))
        summary = json.loads((directory / 'writer-reduction/reduction.json').read_text())
        self.assertEqual(summary['status'], 'complete', summary)
        self.assertTrue(summary['verified'] and summary['deletion_minimal'])
        best = load_writer_plan(directory / 'writer-reduction/best.plan.json')
        self.assertEqual(sum(map(len, best['writers'])), 2, best)
        self.assertEqual(best['writers'][0][0]['edit'], 'w0:2')
        self.assertEqual(plan['writers'], load_writer_plan(directory / 'writer-reduction/original.plan.json')['writers'])
        replay = await self.writer_trial(best, directory / 'replay')
        self.assertEqual(compared(replay), compared(first))
        repaired = copy.deepcopy(best)
        del repaired['test_service_fault']
        passed = await self.writer_trial(repaired, directory / 'correct-service')
        self.assertEqual(passed, dict(outcome='passed', cleanup_complete=True))

    async def test_seeded_writers_resume_queued_edits_through_two_tls_backend_rotations(self):
        self.transport_tls = True
        await self.queued_rotations(rotation_history.generate_queued(self.seed), coalesced=False)

    async def test_seeded_writers_resume_queued_edits_through_two_backend_rotations(self):
        await self.rotation_writers_with_reduction('queued-rotations', rotation_history.generate_queued(self.seed))

    async def test_seeded_coalesced_writers_resume_queued_edits_through_two_backend_rotations(self):
        """The rotation campaign with plain USD authoring, so each writer's queued edits coalesce and chain."""
        await self.rotation_writers_with_reduction('queued-rotations-coalesced',
                                                   rotation_history.generate_queued(self.seed))

    async def rotation_writers_with_reduction(self, kind, plan):
        """Run a root-field rotation plan; a WriterMismatch is reduced through fresh trials and still fails."""
        try:
            await self.exercise_rotation(kind, plan)
        except WriterMismatch as error:
            publish_writer(self.artifact / f'{kind}-mismatch.json', dict(signature=error.signature,
                                                                        evidence=error.evidence))
            # The mismatch counts only once every writer closed cleanly; a crashed writer raises a plain error here,
            # which is not reduced. The mismatch is published first, so it stays on record either way.
            await self.retire_writers()
            # Retire the original fixture before launching any reduction trial.
            await self.cleanup()
            self._cleanups.clear()
            summary = await reduce_writers(plan, error.signature, self.artifact / f'{kind}-reduction',
                                           functools.partial(self.rotation_trial, kind),
                                           domain=round_domain(rotation_history.validate, rotation_history.origin),
                                           context=plan_replay.context(kind))
            print(f'{kind} writer reduction: ' + json.dumps(summary, sort_keys=True), flush=True)
            raise  # Reduction never turns the original failure into a pass.

    async def exercise_rotation(self, kind, plan):
        if kind in ('queued-rotations', 'queued-rotations-coalesced'):
            await self.queued_rotations(plan, coalesced=kind == 'queued-rotations-coalesced')
        else:
            await self.uncertain_writer_rotation(plan, 'before' if kind == 'uncertain-before' else 'after_commit')

    async def rotation_trial(self, kind, plan, directory, *, automatic_reduction=False, crash=False):
        """One fresh fixture, service and set of native writers executing `plan`; the reducer's evaluator. `crash`, a
        test's, kills a writer beside a mismatch (crash_beside_mismatch)."""
        rotation_history.validate(plan)
        directory.mkdir(parents=True, exist_ok=True)
        fixture = PluginTests(ROTATION_TESTS[kind])
        fixture.artifact_override = directory
        fixture.seed_override = plan['master_seed']
        if plan.get('test_service_fault') in CUT_DEFECTS:
            fixture.writer_service_factory = CutWriterService
        elif plan.get('test_service_fault') == 'corrupt-owned-w0:6':
            fixture.writer_service_factory = CorruptRotationWriterService
        if crash:
            return await fresh_trial(fixture, crash_beside_mismatch(fixture, functools.partial(
                fixture.exercise_rotation, kind, plan)))
        return await fresh_trial(fixture, fixture.rotation_writers_with_reduction(kind, plan)
                                 if automatic_reduction else fixture.exercise_rotation(kind, plan))

    async def reduce_rotation_defect(self, kind, plan, signature, units, edits, label=None):
        """Reduce a named test-only defect of a root-field rotation case (reduce_defect); returns the reduction's
        directory."""
        return await self.reduce_defect(functools.partial(self.rotation_trial, kind), rotation_history.validate, kind,
                                        plan, signature, units, edits, label)

    async def reduce_defect(self, trial, validate, kind, plan, signature, units, edits, label=None):
        """Reduce a named test-only defect to its minimal plan through `trial`, the case's evaluator, then replay that
        plan with and without the defect; returns the reduction's directory.

        A trial's evidence depends on the plan, so the replay must match only its outcome and signature."""
        def compared(result):
            return {key: value for key, value in result.items() if key != 'evidence'}
        directory = self.artifact / f'{label or kind}-reducer-selftest'
        first = await trial(plan, directory, automatic_reduction=True)
        self.assertEqual(compared(first), dict(outcome='mismatch', signature=signature, cleanup_complete=True))
        root = directory / f'{kind}-reduction'
        summary = json.loads((root / 'reduction.json').read_text())
        self.assertEqual(summary['status'], 'complete', summary)
        self.assertTrue(summary['verified'] and summary['deletion_minimal'], summary)
        self.assertEqual((summary['retained_units'], summary['retained_edits']), (units, edits))
        self.assertEqual(validate(json.loads((root / 'original.plan.json').read_text())), plan)
        best = validate(json.loads((root / 'best.plan.json').read_text()))
        replay = await trial(best, directory / 'replay')
        self.assertEqual(compared(replay), compared(first))
        repaired = dict(best)
        del repaired['test_service_fault']
        passed = await trial(repaired, directory / 'correct-service')
        self.assertEqual(passed, dict(outcome='passed', cleanup_complete=True))
        return root

    async def test_queued_rotation_fault_reduction_reexecutes_real_clients_and_service(self):
        """The fault-schedule defect forwards the rotation after round 7 to the backend it leaves; the failure keeps
        that rotation, its round and one writer, whose phase after the rotation is empty."""
        plan = dict(rotation_history.generate_queued(self.seed), test_service_fault='misroute-rotation-7')
        await self.reduce_rotation_defect(
            'queued-rotations', plan,
            dict(reason='writers did not resume reading on the rotation destination at its frontier', rotation=7),
            ['rotation:7', 'writer:0', 'round:7'], ['w0:7'])

    async def test_coalesced_queued_rotation_corrupt_reduction_reexecutes_real_clients_and_service(self):
        """Coalesced, the defect commits -1 for writer 0's round-6 edit, an owned field queued during the rotation
        after round 3, in whichever run the writer's window drains; the failure keeps that edit alone."""
        plan = dict(rotation_history.generate_queued(self.seed), test_service_fault='corrupt-owned-w0:6')
        await self.reduce_rotation_defect('queued-rotations-coalesced', plan,
                                          dict(reason='unexplained committed field', field='owned_0_6'),
                                          ['writer:0', 'round:6'], ['w0:6'], label='coalesced-corrupt')

    async def test_coalesced_queued_rotation_fault_reduction_reexecutes_real_clients_and_service(self):
        """Coalesced, the fault-schedule defect forwards the rotation after round 7 to the backend it leaves; the
        failure keeps that rotation, its round and one writer, whose phase after the rotation is empty."""
        plan = dict(rotation_history.generate_queued(self.seed), test_service_fault='misroute-rotation-7')
        await self.reduce_rotation_defect(
            'queued-rotations-coalesced', plan,
            dict(reason='writers did not resume reading on the rotation destination at its frontier', rotation=7),
            ['rotation:7', 'writer:0', 'round:7'], ['w0:7'], label='coalesced-misroute')

    async def test_uncertain_before_cut_reduction_reexecutes_real_clients_and_service(self):
        """The defect commits writer 0's round-0 write when the cut cancels its handler held before execution; the
        failure keeps that write alone. The rotation class's recorded plans replay through the standalone command."""
        plan = dict(rotation_history.generate_uncertain(self.seed), test_service_fault='apply-cut-w0:0')
        root = await self.reduce_rotation_defect('uncertain-before', plan,
                                                 dict(reason='write cut before execution committed', edit='w0:0'),
                                                 ['writer:0', 'round:0'], ['w0:0'])
        await plan_replay.check_standalone(self, root)

    async def test_uncertain_after_cut_reduction_reexecutes_real_clients_and_service(self):
        """The defect passes writer 0's round-0 write through its commit step without committing it; the failure keeps
        that write alone."""
        plan = dict(rotation_history.generate_uncertain(self.seed), test_service_fault='lose-cut-w0:0')
        await self.reduce_rotation_defect('uncertain-after', plan,
                                          dict(reason='write cut after its commit missing from the ledger',
                                               edit='w0:0'),
                                          ['writer:0', 'round:0'], ['w0:0'])

    async def test_uncertain_leaked_cut_reduction_reexecutes_real_clients_and_service(self):
        """The fault-schedule defect releases writer 0's round-0 write, held after its commit, before the cut, so
        writer 0's queued writes reach the service; the failure keeps that write and the first queued behind it."""
        plan = dict(rotation_history.generate_uncertain(self.seed), test_service_fault='leak-cut-w0:0')
        await self.reduce_rotation_defect('uncertain-after', plan,
                                          dict(reason='queued write behind a cut write reached the service',
                                               edit='w0:1'),
                                          ['writer:0', 'round:0', 'round:1'], ['w0:0', 'w0:1'],
                                          label='leaked')

    async def reduce_period_defect(self, kind, plan, signature, units, edits, label):
        """Reduce a named test-only defect of a long-rotations or stream-age plan (reduce_defect), whose first trial
        replays the plan cut after the failing period, never the whole plan."""
        root = await self.reduce_defect(functools.partial(self.period_trial, kind), period_plans.validate, kind, plan,
                                        signature, units, edits, label)
        self.assertTrue(json.loads((root / 'reduction.json').read_text())['truncated'])
        first = json.loads((root / 'trial-0001/input.plan.json').read_text())
        self.assertEqual(period_plans.periods(first), list(range(1, signature['period'] + 1)))
        return root

    async def test_stream_age_corrupt_reduction_truncates_and_reexecutes_real_clients_and_service(self):
        """The defect commits -1 for writer 0's edit of period 5, the period after the first expiry, of 8; the first
        trial runs periods 1 to 5, and the failure keeps that edit alone. The period class's recorded plans replay
        through the standalone command."""
        plan = dict(period_plans.generate(self.seed, 4, 8, period_plans.STREAM_AGE), test_service_fault='corrupt-w0:4')
        root = await self.reduce_period_defect(
            'stream-age', plan, dict(reason='unexplained committed field', field='owned_0_2', period=5),
            ['cycle:2', 'writer:0'], ['w0:4'], 'stream-age-corrupt')
        await plan_replay.check_standalone(self, root)

    async def test_stream_age_resume_reduction_truncates_and_reexecutes_real_clients_and_service(self):
        """The fault-schedule defect commits an unannounced peer edit before the expiry of period 6, of 8, so every
        read resumes past the period's frontier; the first trial runs periods 1 to 6, and the failure keeps that
        expiry's cycle and one writer."""
        plan = dict(period_plans.generate(self.seed, 4, 8, period_plans.STREAM_AGE),
                    test_service_fault='unannounced-peer-6')
        await self.reduce_period_defect(
            'stream-age', plan,
            dict(reason='read did not resume at its coverage frontier after a stream-age expiry', period=6),
            ['cycle:2', 'writer:3'], ['w3:4', 'w3:5'], 'stream-age-resume')

    async def test_period_trial_mismatch_beside_a_crashed_writer_is_unresolved(self):
        """A period evaluator accepts a mismatch only once every writer closed cleanly: beside a writer that exited
        with a failure after its last reply, the same mismatch leaves the trial unresolved."""
        signature = dict(reason='writers did not resume reading on the rotation destination at its frontier', period=6)

        async def trial(crash, directory):
            directory.mkdir(parents=True)
            fixture = PluginTests(PERIOD_TESTS['long-rotations'])
            fixture.artifact_override = directory
            fixture.seed_override = self.seed

            async def run():
                await fixture.start()
                crashed = await fixture.start()
                if crash:
                    # The writer's input ends without close, so it exits 1 after its last reply.
                    crashed.process.stdin.close()
                    await asyncio.wait_for(crashed.process.wait(), 10)
                    self.assertEqual(crashed.process.returncode, 1)
                raise WriterMismatch(signature['reason'], period=signature['period'])
            return await fresh_trial(fixture, run())
        root = self.artifact / 'period-trial-crashed-writer'
        healthy = await trial(False, root / 'healthy')
        self.assertEqual(healthy, dict(outcome='mismatch', signature=signature, cleanup_complete=True))
        crashed = await trial(True, root / 'crashed')
        self.assertEqual((crashed['outcome'], crashed['cleanup_complete'], crashed.get('mismatch')),
                         ('unresolved', True, signature), crashed)
        self.assertNotIn('signature', crashed)

    async def assert_crash_unresolves(self, trial, plan, signature, label):
        """`trial`, a reducer's evaluator, accepts `plan`'s genuine mismatch, and the same mismatch beside a writer
        killed once it is raised (crash_beside_mismatch) leaves the trial unresolved."""
        root = self.artifact / label
        healthy = await trial(plan, root / 'healthy')
        self.assertEqual({key: value for key, value in healthy.items() if key != 'evidence'},
                         dict(outcome='mismatch', signature=signature, cleanup_complete=True))
        crashed = await trial(plan, root / 'crashed', crash=True)
        self.assertEqual((crashed['outcome'], crashed['cleanup_complete'], crashed.get('mismatch')),
                         ('unresolved', True, signature), crashed)
        self.assertNotIn('signature', crashed)
        # The killed writer's exit, not some other retire failure, leaves the trial unresolved.
        self.assertIn(f'{scene_support.KILLED_RETURNCODE} != 0', crashed.get('error', ''), crashed)

    async def test_writer_trial_mismatch_beside_a_crashed_writer_is_unresolved(self):
        """The root-field evaluator accepts a mismatch only once every writer closed cleanly."""
        plan = self.writer_inputs(writer_plan(self.seed, clients=2, steps=4))
        plan['test_service_fault'] = 'corrupt-owned-0-2'
        await self.assert_crash_unresolves(self.writer_trial, plan,
                                           dict(reason='unexplained committed field', field='owned_0_2'),
                                           'writer-trial-crashed-writer')

    async def test_rotation_trial_mismatch_beside_a_crashed_writer_is_unresolved(self):
        """The round-based rotation evaluator accepts a mismatch only once every writer closed cleanly: a queued
        rotation's resume mismatch, raised once its writers converged, and an uncertain cut's, raised while its held
        writes are cut, each over its defect's minimal plan."""
        cases = [
            ('queued-rotations', round_plans.select(dict(rotation_history.generate_queued(self.seed),
                                                         test_service_fault='misroute-rotation-7'), [0], [7], [7]),
             dict(reason='writers did not resume reading on the rotation destination at its frontier', rotation=7)),
            ('uncertain-before', round_plans.select(dict(rotation_history.generate_uncertain(self.seed),
                                                         test_service_fault='apply-cut-w0:0'), [0], [0], []),
             dict(reason='write cut before execution committed', edit='w0:0')),
        ]
        for kind, plan, signature in cases:
            with self.subTest(kind=kind):
                await self.assert_crash_unresolves(functools.partial(self.rotation_trial, kind), plan, signature,
                                                   f'{kind}-trial-crashed-writer')

    async def test_long_rotation_corrupt_reduction_truncates_and_reexecutes_real_clients_and_service(self):
        """The defect commits -1 for writer 0's edit of rotation 5, of 8, on backend B; the first trial runs rotations
        1 to 5, and the failure keeps that edit alone."""
        plan = dict(period_plans.generate(self.seed, 4, 8), test_service_fault='corrupt-w0:4')
        await self.reduce_period_defect(
            'long-rotations', plan, dict(reason='unexplained committed field', field='owned_0_2', period=5),
            ['cycle:2', 'writer:0'], ['w0:4'], 'long-rotations-corrupt')

    async def test_long_rotation_resume_reduction_truncates_and_reexecutes_real_clients_and_service(self):
        """The fault-schedule defect forwards rotation 6, of 8, to backend B, which it leaves, so no writer resumes
        reading on A; the first trial runs rotations 1 to 6, and the failure keeps that rotation's cycle and one
        writer."""
        plan = dict(period_plans.generate(self.seed, 4, 8), test_service_fault='misroute-rotation-6')
        await self.reduce_period_defect(
            'long-rotations', plan,
            dict(reason='writers did not resume reading on the rotation destination at its frontier', period=6),
            ['cycle:2', 'writer:3'], ['w3:4', 'w3:5'], 'long-rotations-resume')

    async def queued_rotations(self, plan, coalesced):
        """Run a queued-rotation plan (rotation_history): its first phase connected, each later one queued during the
        backend rotation before it. Without `coalesced`, `usd-set-admit` binds one intent, and so one chained write,
        to each command; with it, each command is a plain USD change block, so blocks queued during an outage
        coalesce, and each commit is a contiguous run of one writer's edits."""
        rotation_history.validate(plan)
        writers = plan['client_count']
        phases = rotation_history.phases(plan)
        defect = plan.get('test_service_fault')
        prefix = self.write_plan(plan, 'rotation-writers', coalesced_writes=coalesced)
        # Both backends run the fixture's service, so a named service defect fires on whichever backend a selection
        # puts its round on.
        backend = ReferenceServer(self.state, 'B', service_factory=getattr(
            self, 'writer_service_factory', scene_support.CompactSceneService))
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

            async def check_attempts(driver, index, phase, boundary, expected, issuing=False, most=None):
                # While a writer is `issuing` its window, its worker issues each
                # queued write after its local publication, so the count is
                # polled until it reaches the window under a failure-only watchdog.
                # With `most`, `expected` is the least count and `most` the greatest.
                deadline = asyncio.get_running_loop().time() + 5
                while True:
                    snapshot = (await driver.command('diagnostics'))['snapshot']
                    attempts = sum(item['count'] for item in snapshot['metrics']['rpc_attempts_by_operation']
                                   if item['operation'] == 4)  # Public OPERATION_UPDATE.
                    if not issuing or attempts >= expected or asyncio.get_running_loop().time() > deadline:
                        break
                record('client-update-attempts', writer=index, phase=phase, boundary=boundary, count=attempts)
                if most is None:
                    self.assertEqual(attempts, expected, (index, phase, boundary, snapshot))
                else:
                    self.assertTrue(expected <= attempts <= most, (index, phase, boundary, attempts, snapshot))

            record('plan', master_seed=self.seed, scenario=plan['scenario'], writer_seeds=plan['writer_seeds'])
            drivers, wire = [], []
            for _ in plan['writers']:
                reads = len(self.state.scene_reads)
                drivers.append(await self.start(address))
                # A writer reads and writes under one client ID; its first read names it.
                ids = {read['writer'] for read in self.state.scene_reads[reads:]}
                self.assertEqual(len(ids), 1, self.state.scene_reads[reads:])
                wire.append(ids.pop())
            identities = []
            for index, driver in enumerate(drivers):
                identities.append(await driver.command('observe-status'))
                record('writer-ready', writer=index, pid=driver.process.pid, status=identities[-1],
                       client_id=wire[index])
            # Each writer's UpdateLayer calls so far: one per edit bound to its own intent, one per commit coalesced.
            issued = [0] * len(drivers)
            routed = self.address  # The backend the proxy forwards to.

            for phase, (first, end) in enumerate(phases):
                before = len(self.commits())
                read_begin = len(self.state.scene_reads)
                read_faults, write_faults, resumed = [], [], []
                # Rotations alternate A to B and back; the phase after a rotation is queued during it.
                backend = 'A' if phase % 2 == 0 else 'B'
                try:
                    if coalesced and end > first:
                        # Each writer's first write of every phase, the one after
                        # its earlier phases' commits, is held at its commit step;
                        # its chained successors wait.
                        write_faults = [self.state.plan_write('hold', position=issued[index] + 1, writer=client_id)
                                        for index, client_id in enumerate(wire)]
                    if phase:
                        read_attempt = self.state.attempts['SubscribeReadLayer']
                        write_attempt = self.state.attempts['UpdateLayer']
                        read_faults = [self.state.plan('SubscribeReadLayer', 'before_reply',
                                                      attempt=read_attempt + i + 1, pause=True) for i in range(writers)]
                        if not coalesced:
                            # Every queued write of the phase issues within its
                            # writer's window, so each attempt is held before the
                            # service evaluates it.
                            write_faults = [self.state.plan('UpdateLayer', 'before', attempt=write_attempt + i + 1,
                                                            pause=True) for i in range(writers * (end - first))]
                        ordinal = rotation_history.rotation_ordinal(plan, phase - 1)
                        # The test-only fault-schedule defect forwards the rotation after round 7 to the backend it
                        # leaves.
                        destination = other if backend == 'B' else self.address
                        routed = routed if defect == 'misroute-rotation-7' and ordinal == 7 else destination
                        retired = await asyncio.wait_for(proxy.rotate_and_wait(routed, pause=True), 5)
                        self.assertGreaterEqual(retired, writers)
                        record('rotate', phase=phase, rotation=ordinal, backend=backend, boundary=before,
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
                            command = 'usd-set' if coalesced else 'usd-set-admit'
                            result = await driver.command(f"{command} 1 {op['field']} {op['value']}")
                            admitted[index] += 1
                            # Every earlier phase's intents retired at its save.
                            # During an outage the status reports UNAVAILABLE and
                            # nothing retires, so each block bound to its own
                            # intent adds exactly one pending intent, and a
                            # coalesced block at most one; the first phase's
                            # connected updates may retire at any point.
                            if phase and coalesced:
                                self.assertEqual((result['result'], result['status_result']), (0, 8), result)
                                self.assertTrue(1 <= result['pending'] <= admitted[index] - first, result)
                            elif phase:
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
                    if coalesced and not phase:
                        # Connected, each writer's first write reaches the
                        # service while it authors and is held there; nothing
                        # commits until the holds are released.
                        await asyncio.wait_for(asyncio.gather(*(fault.reached.wait() for fault in write_faults)), 5)
                        self.assertEqual(len(self.commits()), before)
                        expected = self.oracle()
                        for index, driver in enumerate(drivers):
                            state = await driver.command('state')
                            own = dict(expected)
                            own.update({op['field']: op['value'] for op in plan['writers'][index][first:end]})
                            self.assertEqual(state['fields'], own)
                            self.assertTrue(1 <= state['pending'] <= end - first, state)
                            record('first-write-held', phase=phase, writer=index, state=state)
                    if phase:
                        expected = self.oracle()
                        for index, driver in enumerate(drivers):
                            state = await driver.command('state')
                            own = dict(expected)
                            own.update({op['field']: op['value'] for op in plan['writers'][index][first:end]})
                            self.assertEqual(state['fields'], own)
                            if coalesced and end > first:
                                # Nothing issues during the outage, so the first
                                # admitted write waits and later blocks coalesce.
                                self.assertTrue(1 <= state['pending'] <= end - first, state)
                                if end - first >= 2:
                                    self.assertLess(state['pending'], end - first, state)
                            else:
                                # One pending intent per queued edit; an empty phase queues none.
                                self.assertEqual(state['pending'], end - first)
                            self.assertEqual(state['subject'], identities[index]['subject'])
                            record('queued-during-outage', phase=phase, writer=index, state=state)
                            await check_attempts(driver, index, phase, 'proxy-paused', issued[index])
                        self.assertEqual(len(self.commits()), before)
                        self.assertEqual(self.state.attempts['UpdateLayer'], write_attempt)
                        proxy.gate.set()
                        await asyncio.wait_for(asyncio.gather(*(fault.reached.wait() for fault in read_faults)), 5)
                        resumed = self.state.scene_reads[read_begin:]
                        self.assertEqual(len(resumed), writers)
                        self.assertEqual(len({read['writer'] for read in resumed}), writers)
                        self.assertTrue(all(read['object'] == 1 for read in resumed), resumed)
                        record('all-caught-up-replies-held', phase=phase, reads=resumed, boundary=before)
                        self.assertEqual(self.state.attempts['UpdateLayer'], write_attempt)
                        for index, driver in enumerate(drivers):
                            await check_attempts(driver, index, phase, 'replies-held', issued[index])
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
                            # Coalesced, the queued blocks make one to all of them.
                            if coalesced and end > first:
                                await check_attempts(driver, index, phase, 'window-issued', issued[index] + 1,
                                                     issuing=True, most=issued[index] + end - first)
                            elif coalesced:
                                # An empty phase, after the last retained round, issues nothing.
                                await check_attempts(driver, index, phase, 'window-issued', issued[index])
                            else:
                                await check_attempts(driver, index, phase, 'window-issued', end, issuing=True)
                finally:
                    proxy.gate.set()
                    for fault in read_faults:
                        fault.release.set()
                    for index in plan['release_orders'][first] if end > first else ():
                        for fault in (write_faults[index:index + 1] if coalesced else write_faults[index::writers]):
                            record('release-write', phase=phase, writer=index, attempt=fault.attempt)
                            fault.release.set()

                # Coalesced, a phase with more blocks than the writer's issuance window leaves the later ones
                # buffered on the host, where a plain save returns BUSY; save-admitted saves again once a flush
                # admits them.
                command = 'save-admitted 5000' if coalesced else 'save 5000'
                saves = await asyncio.gather(*(driver.command(command) for driver in drivers))
                self.assertTrue(all(result['result'] == 0 for result in saves), saves)
                commits = [{'sequence': commit.sequence, 'writer': commit.writer, 'fields': decode_update(commit.payload)}
                           for commit in self.commits()[1:]]
                partial = dict(plan, writers=[writer[:end] for writer in plan['writers']])
                # The accounting comes first, so a lost or repeated commit raises its WriterMismatch.
                expected, lineage = account_writers(partial, admissions, commits, coalesced=coalesced)
                if not coalesced:
                    self.assertEqual(len(self.commits()), 1 + end * writers)
                previous = issued
                issued = [sum(commit['writer'] == client_id for commit in commits) for client_id in wire]
                if coalesced and phase and end - first >= 2:
                    # The outage's coalesced blocks commit in fewer writes than edits.
                    self.assertTrue(all(now - before < end - first for now, before in zip(issued, previous)),
                                    (previous, issued))
                self.assertEqual(expected, self.oracle())
                record('commit-boundary', phase=phase, commits=commits, lineage=lineage, expected=expected)
                for index, driver in enumerate(drivers):
                    state = await self.wait_fields(driver, expected)
                    self.assertEqual((state['result'], state['pending'], state['subject']),
                                     (0, 0, identities[index]['subject']))
                    record('converged', phase=phase, writer=index, state=state)
                    # Coalesced, each of the writer's commits is one of its calls; otherwise each edit is one.
                    await check_attempts(driver, index, phase, 'converged', issued[index] if coalesced else end)
                    self.assertTrue(1 <= issued[index] <= end, (index, issued[index]))
                # No write was re-sent: every UpdateLayer call is one commit.
                self.assertEqual(self.state.attempts['UpdateLayer'], len(commits) if coalesced else end * writers)
                if phase:
                    # The rotation's own check, once its writers converged: each resumed reading on its
                    # destination at the frontier before it.
                    round_plans.check_resume(ordinal, resumed, writers, backend, before)

            if coalesced and 'origin' not in plan:
                # The mode took effect: each writer's writes after an outage
                # were chained behind its held first one, and blocks coalesced.
                # A reduced selection can retain too few queued blocks for
                # either, so this coverage applies to generated plans.
                for index, driver in enumerate(drivers):
                    writes = (await driver.command('diagnostics'))['snapshot']['metrics']['layer_writes']
                    chained = chained_commits(self.state, wire[index])
                    record('layer-writes', writer=index, metrics=writes, chained=len(chained))
                    self.assertTrue(chained, (index, self.state.chain_log))
                    self.assertGreater(writes['window_high_water'], 1, writes)
                    self.assertGreater(writes['coalesced_turns'], 0, writes)
            fresh = await self.start(address)
            await self.wait_fields(fresh, expected)
            self.assertEqual(self.state.attempts['CreateLayer'], 1)
            self.assertEqual(self.state.attempts['SubscribeReadLayer'], writers * len(phases) + 1)
            boundary = len(self.commits())
            closed = await asyncio.gather(*(self.close(driver) for driver in drivers + [fresh]))
            self.assertTrue(all(result['fields'] == expected for result in closed))
            self.assertEqual(len(self.commits()), boundary, 'teardown issued another mutation')
            if not coalesced:
                self.assertEqual(boundary, 1 + writers * plan['rounds'])
            record('complete', admitted=len(admissions), boundary=boundary, rotations=len(phases) - 1)

    def write_plan(self, plan, label, **fields):
        """Record the observed workload and write the plan artifact before any writer starts; returns its prefix.

        The artifact, also the trace's `writer_plan`, is `plan` with the fixture and candidate identities and
        `fields`; `plan` itself is unchanged, so a reduction trial replays exactly its inputs.
        """
        campaign.observe_workload(self, plan)
        candidate = json.loads(Path(os.environ['LIVE_TEST_PLUGIN_DRIVER'] + '.manifest.json').read_text())
        artifact = dict(plan, initial_fixture_sha256=hashlib.sha256(empty_snapshot()).hexdigest(),
                        candidate=candidate, **fields)
        prefix = self.artifact / self.id().split('.')[-1]
        Path(str(prefix) + '.plan.json').write_text(json.dumps(artifact, sort_keys=True) + '\n')
        print(f"{label} master_seed={self.seed} clients={plan['client_count']} rounds={len(plan['writers'][0])} "
              f"plan={prefix}.plan.json", flush=True)
        self.trace['writer_plan'] = artifact
        return prefix

    async def rotation_setup(self, plan, label, backend_factory=scene_support.CompactSceneService):
        """Write the plan artifact, then start backend B, running `backend_factory`'s service, and the RotatingProxy in
        front of A."""
        prefix = self.write_plan(plan, label)
        backend = ReferenceServer(self.state, 'B', service_factory=backend_factory)
        self.extra_servers.append(backend)
        backends = {'A': self.address, 'B': await backend.start()}
        proxy = scene_support.RotatingProxy(self.address)
        self.proxies.append(proxy)
        return prefix, proxy, backends, await proxy.start()

    async def test_seeded_writers_converge_through_many_backend_rotations(self):
        """Separated rotations never accumulate toward the reconnect attempt limit.

        Each rotation parks new connections, so every writer's read resumes
        once, queues one edit without issuing it, recovers on the other backend
        and converges before the next rotation. Each rotation's first read
        retry is scheduled at the first backoff step, which shows the reconnect
        attempt count resets at every recovery readiness; a retry that fails
        before reaching a backend is the next attempt, in its own window
        (period_plans.check_rotation_reconnects). The caught-up reply gate is
        armed only in every 50th seeded period and in the generated plan's
        last (long_rotations), because the service scans every planned fault
        on each RPC.

        Each rotation's ledger accounting and resume check raise a
        WriterMismatch naming the seeded period, which is reduced
        (period_with_reduction); every other check raises a plain
        error.
        """
        await self.period_with_reduction('long-rotations', rotation_writer_plan(
            self.seed, getattr(self, 'writer_clients', 4), getattr(self, 'writer_steps', 8)))

    async def long_rotations(self, plan):
        """Run a long-rotations period plan (period_plans), a generated one or a selection of one.

        Rotation r of the run is the plan's position r - 1 and runs its seeded period: the period's parity names the
        destination backend, and the caught-up reply gate is armed in every 50th seeded period and in the generated
        plan's last, so a plan cut after a period runs exactly the inputs the generated plan ran through it. Every
        count is of the rotations run.
        """
        period_plans.validate(plan)
        clients = plan['client_count']
        planned = period_plans.periods(plan)
        rotations, last = len(planned), period_plans.last_period(plan)
        defect = plan.get('test_service_fault')
        # The test-only defect corrupt-w0:4 corrupts writer 0's ordinal-4 edit, when the plan holds it; backend B
        # runs the fixture's service too, so the defect fires on whichever backend a selection puts it.
        self.state.corrupt_item = next(((op['field'], op['value']) for writer in plan['writers'] for op in writer
                                        if op['edit'] == 'w0:4'), None) if defect == 'corrupt-w0:4' else None
        prefix, proxy, backends, address = await self.rotation_setup(
            plan, 'long-rotation-writers', getattr(self, 'writer_service_factory', scene_support.CompactSceneService))
        with Path(str(prefix) + '.history.jsonl').open('w') as history:
            run = RotationRun(self, plan, proxy, backends, history)
            run.record('plan', master_seed=self.seed, scenario=plan['scenario'], writer_seeds=plan['writer_seeds'],
                       clients=clients, rotations=rotations, periods=planned)
            await run.join([await self.start(address) for _ in plan['writers']])
            for rotation, period in enumerate(planned, 1):
                # The test-only fault-schedule defect forwards rotation 6 to the backend it leaves.
                misrouted = defect == 'misroute-rotation-6' and period == 6
                fields = await run.rotate(rotation, period=period, gated=period % 50 == 0 or period == last,
                                          destination=('A' if period % 2 else 'B') if misrouted else None)
                self.assertEqual(self.state.attempts['CreateLayer'], 1)
                self.assertEqual(self.state.attempts.get('UpdateLayer', 0), clients * rotation)
                self.assertEqual(self.state.attempts['SubscribeReadLayer'], clients * (1 + rotation))
                run.record('rotation', rotation=rotation, period=period, commits=len(self.commits()),
                           fields=run.fields_digest(),
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
                snapshots = [await settled_snapshot(driver) for driver in run.drivers]
                for snapshot in snapshots:
                    auth = snapshot['metrics']['auth']
                    self.assertEqual(auth['failures_by_cause'], [], auth)
                    self.assertTrue(all(entry['resolution'] == 0 for entry in auth['credential_wait_ms_by_resolution']['entries']),
                                    auth)
                    self.assertLessEqual(set(ping_results(snapshot)), {0, 8, 10}, ping_results(snapshot))
                # 9. Process figures and retained-work gauges.
                samples = [soak_sample(driver.process.pid, snapshot) for driver, snapshot in zip(run.drivers, snapshots)]
                for index, sample in enumerate(samples):
                    series[index].append((rotation, sample['anon'], sample['threads']))
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

            # The second half of each writer's series: its whole least-squares slope, as a figure, and its growth
            # checked per run of unchanged thread count (campaign.rss_growth).
            halves = [[point for point in points if point[0] >= rotations // 2] for points in series]
            slopes = [campaign.rss_slope([(step, value) for step, value, _ in half]) for half in halves]
            growth = [campaign.rss_growth(half, base['threads']) for half, base in zip(halves, baseline or [])]
            challenges = [{reason: sum(entry[1] == reason for entry in log) for reason in (1, 2, 3)} for log in auth_log]
            summary = dict(
                schema_version=1, master_seed=self.seed, clients=clients, rotations=rotations,
                wall_time_s=round(time.monotonic() - started, 3), scale=scale, holds=holds,
                read_events=read_events, update_events=update_events, extra_commits=extra,
                rejections=dict(read=sum(rejections('SubscribeReadLayer')), update=sum(rejections('UpdateLayer'))),
                commits=len(self.commits()), attempts=dict(self.state.attempts),
                challenges=[{'missing': c[1], 'expiring': c[2], 'rejected': c[3]} for c in challenges],
                generations=current, pings_ok=pings, read_generations=run.generations,
                baseline=baseline, final=samples, anon_slope_bytes_per_rotation=slopes, anon_growth=growth,
                thresholds=dict(rss_ratio=campaign.SOAK_RSS_RATIO, rss_slack_bytes=campaign.SOAK_RSS_SLACK_BYTES,
                                rss_slope_bytes=campaign.SOAK_RSS_SLOPE_BYTES, extra_threads=campaign.SOAK_EXTRA_THREADS,
                                rss_min_segment=campaign.SOAK_RSS_MIN_SEGMENT,
                                rss_thread_bytes=campaign.SOAK_RSS_THREAD_BYTES,
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
                    self.assertEqual(growth[index]['failures'], [], (index, growth[index]))

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

    async def test_seeded_writers_and_joins_survive_stream_age_expiries_after_server_token_expiry(self):
        """Long sessions behind an ingress that limits stream age, compressed in time.

        Every writer reads the layer and joins its session through an HTTP/2
        proxy that resets each stream once it reaches campaign.STREAM_AGE_MAX_MS
        on the proxy's stream clock, the 240-minute ingress stream timeout the
        deployment research records. Each period advances that clock by half the
        limit, so every second period ends every writer's read and join on its
        open connection, and the periods between show that the proxy ends no
        younger stream. Before each such expiry the service stops accepting
        every token the writer has held, as a token lifetime far shorter than the
        stream age does, so every ended stream outlived its token; proactive
        refresh over wall time is the soak case's. Each period every writer authors one planned edit and
        sends one presence message, and an independent peer commits; the period
        ends with exact accounting of commits, attempts and rejections, every
        message delivered once to every other writer, every writer converged,
        and every retained-work gauge, thread and open file at its warm-up
        value. The scale comes from writer_stream_age: the campaign cases set
        campaign.stream_age_scale(); plain discovery uses
        campaign.STREAM_AGE_HARNESS, which runs every path and asserts the
        memory bounds only at campaign scale.

        The ledger accounting of each period's writer commits and each expiry's
        resume check raise a WriterMismatch naming the seeded period, which is
        reduced (period_with_reduction); every other check raises a plain
        error.
        """
        await self.period_with_reduction('stream-age', period_plans.generate(
            self.seed, getattr(self, 'writer_clients', 4),
            getattr(self, 'writer_steps', campaign.STREAM_AGE_HARNESS_PERIODS), period_plans.STREAM_AGE))

    async def exercise_period(self, kind, plan):
        await (self.long_rotations(plan) if kind == 'long-rotations' else self.stream_age(plan))

    async def period_with_reduction(self, kind, plan):
        """Run a long-rotations or stream-age period plan; a WriterMismatch is reduced through fresh trials, the first
        replaying the plan cut after the failing period, and still fails."""
        try:
            await self.exercise_period(kind, plan)
        except WriterMismatch as error:
            publish_writer(self.artifact / f'{kind}-mismatch.json', dict(signature=error.signature,
                                                                        evidence=error.evidence))
            # The mismatch counts only once every writer closed cleanly; a crashed writer raises a plain error here,
            # which is not reduced. The mismatch is published first, so it stays on record either way.
            await self.retire_writers()
            # Retire the original fixture before launching any reduction trial.
            await self.cleanup()
            self._cleanups.clear()
            # A stream-age trial runs at this case's scale, so a replay of its plans does too.
            scale = getattr(self, 'writer_stream_age', campaign.STREAM_AGE_HARNESS) if kind == 'stream-age' else None
            summary = await reduce_writers(plan, error.signature, self.artifact / f'{kind}-reduction',
                                           functools.partial(self.period_trial, kind), domain=period_domain(),
                                           context=plan_replay.context(kind, scale))
            print(f'{kind} writer reduction: ' + json.dumps(summary, sort_keys=True), flush=True)
            raise  # Reduction never turns the original failure into a pass.

    async def period_trial(self, kind, plan, directory, *, automatic_reduction=False):
        """One fresh fixture, service, proxy and set of native writers executing `plan` at this case's scale; the
        reducer's evaluator."""
        period_plans.validate(plan)
        directory.mkdir(parents=True, exist_ok=True)
        fixture = PluginTests(PERIOD_TESTS[kind])
        fixture.artifact_override = directory
        fixture.seed_override = plan['master_seed']
        fixture.writer_stream_age = getattr(self, 'writer_stream_age', campaign.STREAM_AGE_HARNESS)
        if plan.get('test_service_fault') == 'corrupt-w0:4':
            fixture.writer_service_factory = CorruptPeriodWriterService
        return await fresh_trial(fixture, fixture.period_with_reduction(kind, plan) if automatic_reduction else
                                 fixture.exercise_period(kind, plan))

    async def stream_age(self, plan):
        """Run a stream-age period plan (period_plans), a generated one or a selection of one.

        Period p runs the plan's ordinal p - 1; whether it ends every stream follows its seeded parity, and every
        count of periods, expiries and commits is of the periods run.
        """
        from http2_proxy import Http2Proxy
        period_plans.validate(plan)
        clients = plan['client_count']
        planned = period_plans.periods(plan)
        periods = len(planned)
        scale = getattr(self, 'writer_stream_age', campaign.STREAM_AGE_HARNESS)
        defect = plan.get('test_service_fault')
        # The test-only defect corrupt-w0:4 corrupts writer 0's ordinal-4 edit, when the plan holds it.
        self.state.corrupt_item = next(((op['field'], op['value']) for writer in plan['writers'] for op in writer
                                        if op['edit'] == 'w0:4'), None) if defect == 'corrupt-w0:4' else None
        tokens = {}
        for index in range(clients):
            for generation in range(1, campaign.STREAM_AGE_MAX_GENERATIONS + 1):
                name = f'age-w{index}-{generation}'
                self.state.issue(name, 'writer')
                tokens[hashlib.sha256(name.encode()).hexdigest()] = (index, generation)
        prefix = self.write_plan(plan, 'stream-age-writers')
        proxy = Http2Proxy(self.address, max_stream_age_ms=campaign.STREAM_AGE_MAX_MS, record_frames=False)
        self.proxies.append(proxy)
        address = await proxy.start()
        started = time.monotonic()
        peers, expiries, done = campaign.STREAM_AGE_PEER_EDITS, 0, 0
        auth_log, current, expired = [[] for _ in range(clients)], [1] * clients, [0] * clients
        remembered, baseline, samples, series = [], None, [], [[] for _ in range(clients)]
        connections, received = [None] * clients, [0] * clients

        def rejections(method):
            """Rejected attempts per writer for one method, each naming a writer token."""
            counts = [0] * clients
            for rejection in self.state.auth_rejections:
                if rejection['method'] == method:
                    counts[tokens[rejection['token_sha256']][0]] += 1
            return counts

        with Path(str(prefix) + '.history.jsonl').open('w') as history:
            run = RotationRun(self, plan, proxy, None, history)

            async def auth_events(index):
                reply = await run.drivers[index].command('auth-events')
                auth_log[index].extend(reply['events'])
                current[index] = reply['generation']
                return reply

            async def expire_held():
                """The service stops accepting every token each writer has held."""
                for index in range(clients):
                    await auth_events(index)
                    for generation in range(expired[index] + 1, current[index] + 1):
                        self.state.expire(f'age-w{index}-{generation}')
                    expired[index] = current[index]
                    remembered.append((index, current[index]))

            async def presence(index, connected):
                """The writer's presence events once its join connected `connected` times and received every message."""
                reply = await run.drivers[index].command(f'session-events {connected} {received[index]}')
                self.assertEqual((reply['state'], reply['result']), (2, 0), reply)
                return reply['events']

            async def resumed(index, driver):
                """The writer's read records up to its next stream_resumed, under a failure-only watchdog."""
                records, deadline = [], time.monotonic() + 15
                while not any(record['event_name'] == 'stream_resumed' for record in records):
                    self.assertLess(time.monotonic(), deadline, records)
                    records += (await run.read_records(index, driver))[0]
                return records

            run.record('plan', master_seed=self.seed, scenario=plan['scenario'],
                       writer_seeds=plan['writer_seeds'], clients=clients, periods=periods, scale=scale,
                       max_stream_age_ms=campaign.STREAM_AGE_MAX_MS, step_ms=campaign.STREAM_AGE_STEP_MS,
                       peer_edits=peers)
            await run.join([await self.start(address, environment=dict(
                LIVE_TEST_TOKEN_PREFIX=f'age-w{index}')) for index in range(clients)])
            for index, driver in enumerate(run.drivers):
                self.assertEqual((await driver.command(f'session-join w{index}'))['result'], 0)
            for index in range(clients):
                connected = [event for event in await presence(index, 1) if event[0] == 1]
                self.assertEqual([event[1] for event in connected], [0], connected)
                connections[index] = connected[0][2]
            self.assertEqual(len(set(connections)), clients)
            for position, period in enumerate(planned):
                done += 1
                expiry = period % 2 == 0
                pings = [ping_results((await driver.command('diagnostics'))['snapshot']).get(0, 0)
                         for driver in run.drivers]
                boundary, joins_rejected = len(self.commits()), rejections('JoinSession')
                # 1. Every token each writer has held expires before its streams reach the age limit.
                if expiry:
                    await expire_held()
                if defect == 'unannounced-peer-6' and period == 6:
                    # The test-only fault-schedule defect commits a peer edit that the period's frontier does not
                    # count, and every writer applies it before its streams end.
                    await self.peer({'unannounced': -1})
                    for driver in run.drivers:
                        await self.wait_fields(driver, dict(run.expected, unannounced=-1))
                # 2. The stream clock advances; at the limit the ingress ends each writer's read and join.
                ended = await proxy.advance_stream_clock(campaign.STREAM_AGE_STEP_MS)
                if not expiry:
                    self.assertEqual(ended, [])
                else:
                    expiries += 1
                    streams = {}
                    for record in ended:
                        streams.setdefault(record['connection'], []).append(record['age_ms'])
                    self.assertEqual(sorted(streams.values()), [[campaign.STREAM_AGE_MAX_MS] * 2] * clients, ended)
                    # 3. Every read resumes once, from its coverage frontier, on the same connection: the expiry's
                    # own check of each writer's first resume, decided over every writer once each has resumed; a
                    # second resume fails a plain check here or at step 10, whichever read it lands in.
                    records = [[r['fields'] for r in await resumed(index, driver) if r['event_name'] == 'stream_resumed']
                               for index, driver in enumerate(run.drivers)]
                    period_plans.check_expiry_resume(period, [writer[0]['writer'] for writer in plan['writers']],
                                                     records, run.generations, boundary)
                    run.generations = [resumed_reads[0]['stream_generation'] for resumed_reads in records]
                    # 4. Every join rejoins as a new connection: the ended stream reports CANCELLED and each
                    # attempt the service rejected UNAUTHENTICATED.
                    for index in range(clients):
                        connected = [event for event in await presence(index, expiries + 1) if event[0] == 1]
                        rejected = rejections('JoinSession')[index] - joins_rejected[index]
                        results = [event[1] for event in connected]
                        self.assertEqual((results[0], results[-1]), (10, 0), connected)
                        self.assertEqual(sorted(results[1:-1]), [6] * rejected, connected)
                        self.assertNotIn(connected[-1][2], connections, connected)
                        connections[index] = connected[-1][2]
                    # 5. Every read generation pings: each writer finishes a Ping OK after its reconnect, under a
                    # failure-only watchdog.
                    for index, driver in enumerate(run.drivers):
                        async def pinged():
                            while True:
                                count = ping_results((await driver.command('diagnostics'))['snapshot']).get(0, 0)
                                if count > pings[index]:
                                    return count
                        pings[index] = await asyncio.wait_for(pinged(), 15)
                # 6. One planned edit per writer, each its own commit.
                ops = [writer[position] for writer in plan['writers']]
                for index, driver in enumerate(run.drivers):
                    result = await driver.command(f"usd-set 1 {ops[index]['field']} {ops[index]['value']}")
                    self.assertEqual(result['result'], 0, result)
                saves = await writer_tasks(*(driver.command('save 5000') for driver in run.drivers))
                self.assertTrue(all(result['result'] == 0 for result in saves), saves)
                commits = self.commits()[boundary:]
                decoded = [{'sequence': commit.sequence, 'writer': commit.writer, 'fields': decode_update(commit.payload)}
                           for commit in commits]
                # The period's ledger accounting names the period, which the reducer's truncation cuts after.
                with period_plans.in_period(period):
                    changed, _ = account_writers(dict(plan, writers=[[op] for op in ops]),
                                                 [{'edit': op['edit']} for op in ops], decoded, base_sequence=boundary)
                owners = {(op['field'], op['value']): op['writer'] for op in ops}
                for commit, entry in zip(commits, decoded):
                    writer = owners[next(iter(entry['fields'].items()))]
                    self.assertEqual(run.identities.setdefault(writer, commit.writer), commit.writer,
                                     'writer identity changed across stream-age expiries')
                    run.model.apply(commit.payload, commit.sequence)
                run.expected.update(changed)
                # 7. Independent peer commits.
                for ordinal in range(peers):
                    edit = {f'peer_{(2 * period + ordinal) % 3}': -(10000 * period + ordinal)}
                    before = len(self.commits())
                    await self.peer(edit)
                    commit = self.commits()[before]
                    self.assertEqual((len(self.commits()), commit.writer, decode_update(commit.payload)),
                                     (before + 1, 'independent-peer', edit))
                    run.model.apply(commit.payload, commit.sequence)
                    run.expected.update(edit)
                # 8. One message per writer, delivered once to every other writer from its current connection.
                for index, driver in enumerate(run.drivers):
                    self.assertEqual((await driver.command(f'session-send p{period}w{index}'))['result'], 0)
                for index in range(clients):
                    received[index] += clients - 1
                    events = await presence(index, expiries + 1)
                    self.assertEqual([event for event in events if event[0] == 1], [], events)
                    self.assertEqual(sorted((event[3], event[2]) for event in events if event[0] == 4),
                                     sorted((f'p{period}w{other}', connections[other])
                                            for other in range(clients) if other != index), events)
                # 9. Convergence: nothing pending and every writer holds the authoritative fields.
                self.assertEqual(run.expected, run.model.state)
                for driver in run.drivers:
                    self.assertEqual((await self.wait_fields(driver, run.expected))['pending'], 0)
                # 10. Exact accounting, adjusted by the rejected attempts; no read reconnected outside an expiry.
                rejected = {method: rejections(method) for method in ('SubscribeReadLayer', 'JoinSession', 'UpdateLayer')}
                self.assertEqual(self.state.attempts['CreateLayer'], 1)
                self.assertEqual(self.state.attempts['UpdateLayer'],
                                 (clients + peers) * done + sum(rejected['UpdateLayer']))
                self.assertEqual(self.state.attempts['SubscribeReadLayer'],
                                 clients * (1 + expiries) + sum(rejected['SubscribeReadLayer']))
                self.assertEqual(self.state.attempts['JoinSession'], clients * (1 + expiries) + sum(rejected['JoinSession']))
                self.assertEqual(len(self.commits()), 1 + (clients + peers) * done)
                for index, driver in enumerate(run.drivers):
                    reads, attempts = await run.read_records(index, driver)
                    self.assertEqual(attempts, done + rejected['UpdateLayer'][index])
                    self.assertFalse([r for r in reads if r['event_name'] == 'stream_resumed'], reads)
                # The service rejected only expired tokens, and every generation a writer held when it expired
                # was rejected.
                named = [tokens[rejection['token_sha256']] for rejection in self.state.auth_rejections]
                self.assertTrue(all(rejection['reason'] == 'expired' for rejection in self.state.auth_rejections),
                                self.state.auth_rejections)
                self.assertTrue(all(generation <= expired[index] for index, generation in named), (named, expired))
                for index in range(clients):
                    await auth_events(index)
                    challenges = [entry[0] for entry in auth_log[index]]
                    self.assertEqual(len(challenges), len(set(challenges)), auth_log[index])
                    self.assertTrue(all(entry[5] in (0, 15) for entry in auth_log[index]), auth_log[index])
                for index, generation in remembered:
                    self.assertIn((index, generation), named, (named, auth_log[index]))
                # 11. No auth failure, every credential wait resolved usable, and Ping attempts end OK,
                # UNAVAILABLE, CANCELLED or, once per rejection, UNAUTHENTICATED.
                snapshots = [await settled_snapshot(driver) for driver in run.drivers]
                ping_rejected = rejections('Ping')
                for index, snapshot in enumerate(snapshots):
                    auth = snapshot['metrics']['auth']
                    self.assertEqual(auth['failures_by_cause'], [], auth)
                    self.assertTrue(all(entry['resolution'] == 0
                                        for entry in auth['credential_wait_ms_by_resolution']['entries']), auth)
                    results = ping_results(snapshot)
                    self.assertLessEqual(set(results), {0, 6, 8, 10}, results)
                    self.assertEqual(results.get(6, 0), ping_rejected[index], results)
                # 12. Process figures and retained-work gauges; after warm-up every gauge is at its baseline value.
                samples = [soak_sample(driver.process.pid, snapshot) for driver, snapshot in zip(run.drivers, snapshots)]
                for index, sample in enumerate(samples):
                    series[index].append((done, sample['anon'], sample['threads']))
                if baseline is None and done >= scale['warmup']:
                    baseline = samples
                drift = [soak_drift(base, sample) for base, sample in zip(baseline, samples)] if baseline else []
                run.record('period', period=period, expiry=expiry, ended=len(ended), commits=len(self.commits()),
                           fields=run.fields_digest(), generations=list(run.generations), connections=list(connections),
                           auth_generations=list(current), pings=pings,
                           rejected={method: sum(counts) for method, counts in rejected.items()},
                           samples=samples, drift=drift)
                if baseline:
                    for index, (base, sample) in enumerate(zip(baseline, samples)):
                        self.assertEqual(drift[index], {}, (period, index))
                        self.assertLessEqual(sample['threads'], base['threads'] + campaign.SOAK_EXTRA_THREADS,
                                             (period, index, base['threads'], sample['threads']))
                        self.assertLessEqual(sample.get('timer_threads', 0), campaign.SOAK_MAX_GRPC_TIMER_THREADS,
                                             (period, index, sample.get('timer_threads')))
                        self.assertLessEqual(sample['files'], base['files'] + campaign.SOAK_EXTRA_FILES,
                                             (period, index, base['files'], sample['files']))

            # The second half of each writer's series: its whole least-squares slope, as a figure, and its growth
            # checked per run of unchanged thread count (campaign.rss_growth).
            halves = [[point for point in points if point[0] >= periods // 2] for points in series]
            slopes = [campaign.rss_slope([(step, value) for step, value, _ in half]) if half else 0.0
                      for half in halves]
            growth = [campaign.rss_growth(half, base['threads']) for half, base in zip(halves, baseline or [])]
            challenges = [{reason: sum(entry[1] == reason for entry in log) for reason in (1, 2, 3)} for log in auth_log]
            summary = dict(
                schema_version=1, master_seed=self.seed, clients=clients, periods=periods, expiries=expiries,
                wall_time_s=round(time.monotonic() - started, 3), scale=scale,
                stream_clock_ms=proxy.stream_clock_ms, max_stream_age_ms=campaign.STREAM_AGE_MAX_MS,
                rejections={method: sum(rejections(method)) for method in
                            ('SubscribeReadLayer', 'JoinSession', 'UpdateLayer', 'Ping')},
                commits=len(self.commits()), attempts=dict(self.state.attempts),
                challenges=[{'missing': c[1], 'expiring': c[2], 'rejected': c[3]} for c in challenges],
                generations=current, read_generations=run.generations, baseline=baseline, final=samples,
                anon_slope_bytes_per_period=slopes, anon_growth=growth)
            Path(str(prefix) + '.stream-age-summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
            print('stream-age-summary ' + json.dumps(dict(periods=periods, expiries=expiries,
                                                          wall_time_s=summary['wall_time_s'], generations=current,
                                                          anon=[sample['anon'] for sample in samples], slopes=slopes)),
                  flush=True)
            self.assertEqual(expiries, sum(period % 2 == 0 for period in planned))
            self.assertTrue(all(generation < campaign.STREAM_AGE_MAX_GENERATIONS for generation in current), current)
            # The warm-up baseline and the memory bounds apply to generated plans: a reduced selection can end
            # before the case's warm-up.
            if 'origin' not in plan:
                self.assertIsNotNone(baseline)
            if scale['thresholds'] and 'origin' not in plan:
                bound = lambda value: campaign.SOAK_RSS_RATIO * value + campaign.SOAK_RSS_SLACK_BYTES
                for index, (base, sample) in enumerate(zip(baseline, samples)):
                    self.assertLessEqual(sample['anon'], bound(base['anon']), (index, base['anon'], sample['anon']))
                    self.assertLessEqual(sample['hwm'], bound(base['hwm']), (index, base['hwm'], sample['hwm']))
                    if scale['slope']:
                        self.assertEqual(growth[index]['failures'], [], (index, growth[index]))

            fresh = await self.start(address)
            state = await self.wait_fields(fresh, run.expected)
            self.assertEqual(state['pending'], 0, state)
            self.assertEqual(await client_update_attempts(fresh), 0)
            run.record('fresh-observer-converged', fields=len(run.expected))
            closed = await writer_tasks(*(self.close(driver) for driver in run.drivers + [fresh]))
            self.assertTrue(all(result['fields'] == run.expected for result in closed))
            boundary = 1 + (clients + peers) * periods
            self.assertEqual(len(self.commits()), boundary, 'teardown issued another mutation')
            self.assertEqual(run.expected, self.oracle())
            # Every join left: the session holds no participant.
            self.assertEqual(self.state.sessions[self.session].peers, {})
            run.record('complete', periods=periods, expiries=expiries, boundary=boundary,
                       wall_time_s=summary['wall_time_s'])

    async def test_rotation_before_execution_retains_uncertain_and_queued_writer_intents(self):
        await self.rotation_writers_with_reduction('uncertain-before', rotation_history.generate_uncertain(self.seed))

    async def test_rotation_after_commit_retains_uncertain_and_queued_writer_intents(self):
        await self.rotation_writers_with_reduction('uncertain-after', rotation_history.generate_uncertain(self.seed))

    async def settled_cut(self, drivers, faults, base, ended):
        """Wait until the uncertain cut's ledger is final: every UpdateLayer call the writers report issuing has
        reached the service and every one not held at an unreleased fault has ended.

        `base` and `ended` are the service's UpdateLayer attempts and ended calls before the writers wrote. The
        service's call ends drive the wait; the timeout is only a failure watchdog, which leaves a plain error.
        """
        issued = sum([await client_update_attempts(driver) for driver in drivers])

        def final():
            arrived = self.state.attempts.get('UpdateLayer', 0) - base
            in_flight = arrived - (self.state.ended['UpdateLayer'] - ended)
            return arrived >= issued and in_flight == sum(not fault.release.is_set() for fault in faults)
        async with asyncio.timeout(5):
            await self.state.observed(final)

    async def uncertain_writer_rotation(self, plan, boundary):
        """Run an uncertain-write plan (rotation_history): every writer's first write is held at `boundary`, its
        other edits are queued behind it, and a backend rotation cuts the held writes."""
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        rotation_history.validate(plan)
        writers = plan['client_count']
        edits = sum(map(len, plan['writers']))
        prefix = self.write_plan(plan, f'uncertain-writers boundary={boundary}', fault_boundary=boundary)
        defect = plan.get('test_service_fault')
        target = next((op for ops in plan['writers'] for op in ops if op['edit'] == 'w0:0'), None)
        if defect in CUT_DEFECTS and target:
            # The test-only defect acts on writer 0's round-0 write while the plan retains it.
            self.state.cut_defect = (CUT_DEFECTS[defect], {target['field']: target['value']})
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

            def cut_commits():
                # The writer commits after the fixture's CreateLayer and the peer write.
                return [dict(sequence=entry.sequence, fields=decode_update(entry.payload))
                        for entry in self.commits()[2:]]
            base, ended = self.state.attempts['UpdateLayer'], self.state.ended['UpdateLayer']
            faults = [self.state.plan('UpdateLayer', boundary, attempt=base + i + 1, pause=True)
                      for i in range(writers)]
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
                if getattr(self.state, 'cut_defect', (None,))[0] == 'leak' and len(plan['writers'][0]) > 1:
                    # The test-only fault-schedule defect releases writer 0's round-0 write, held after its commit,
                    # while writer 0 has queued writes, and waits until they have followed it into the ledger.
                    next(f for f in faults if f.attempt == self.state.cut_attempt).release.set()
                    leaked = {(op['field'], op['value']) for op in plan['writers'][0]}
                    async with asyncio.timeout(5):
                        await self.state.observed(lambda: leaked <= {item for entry in cut_commits()
                                                                     for item in entry['fields'].items()})
                await self.settled_cut(drivers, faults, base, ended)
                # The ledger is final until the cut, so the cut oracle classifies it before the attempt and pending
                # checks, which an issued queued write would also fail.
                rotation_history.check_cut(plan, boundary, cut_commits())
                for index, driver in enumerate(drivers):
                    await attempts(driver, index, 'first-rpc-held')
                    state = await driver.command('state')
                    self.assertEqual(state['pending'], len(plan['writers'][index]))
                    self.assertEqual(state['fields']['shared_peer'], peer_write['shared_peer'])
                    record('before-cut', writer=index, state=state)
                    before_cut.append(state['fields'])
                self.assertEqual(self.state.attempts['UpdateLayer'], base + writers)
                committed = writers if boundary == 'after_commit' else 0
                self.assertEqual(len(self.commits()), committed + 2)
                retired = await asyncio.wait_for(proxy.rotate_and_wait(other, pause=True), 5)
                self.assertGreaterEqual(retired, writers)
                # Transport retirement alone cannot prove the old service handler
                # stopped before execution. Each paused handler must see actual
                # cancellation while its release event remains unset.
                await asyncio.wait_for(asyncio.gather(*(fault.cancelled.wait() for fault in faults)), 5)
                self.assertTrue(all(not fault.release.is_set() for fault in faults))
                record('cut-and-service-cancelled', backend='B', retired_forwards=retired,
                       cancelled_attempts=[fault.attempt for fault in faults], committed=committed)
                # Every cut handler has ended, so the ledger is final.
                rotation_history.check_cut(plan, boundary, cut_commits())
                self.assertEqual(len(self.commits()), committed + 2)
                for index, driver in enumerate(drivers):
                    status = await driver.command('await-status 7')
                    self.assertEqual(status['subject'], identities[index]['subject'])
                    self.assertGreater(status['incident'], 0)
                    state = await driver.command('state')
                    self.assertEqual((state['state'], state['result'], state['pending']),
                                     (7, 19, len(plan['writers'][index])))
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
            # Actual commit bytes account for precisely the first operation from
            # every writer; queued suffixes must never appear on the service.
            # Two first operations may write one shared field, which the later
            # commit wins.
            planned_first = {}
            if committed:
                partial = dict(plan, writers=[ops[:1] for ops in plan['writers']])
                planned_first, _ = account_writers(partial, [items[0] for items in admissions], [
                    dict(sequence=entry.sequence, writer=entry.writer, fields=decode_update(entry.payload))
                    for entry in self.commits()[2:]], base_sequence=self.commits()[1].sequence)
            self.assertEqual(self.oracle(), dict(peer_write, **planned_first))
            for index, (driver, expected) in enumerate(zip(drivers, clients)):
                self.assertEqual((await driver.command('save 5000'))['result'], 19)
                path = Path(str(prefix) + f'.writer-{index}.usda')
                exported = await export_layer(driver, path)
                self.assertEqual(exported['result'], 0, exported)
                state = await driver.command('state')
                # FAILED latched at await-status 7, so the host content is
                # already frozen there; SdfLayer::Export writes exactly it.
                self.assertEqual(state['fields'], frozen[index])
                self.assertEqual(exported['fields'], state['fields'])
                expected.update(export=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), state=state)
                await attempts(driver, index, 'exported')
                record('exported', writer=index, **expected)
            self.assertEqual(self.state.attempts['UpdateLayer'], base + writers)
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
            self.assertEqual(report['verified_operations'], edits)
            record('offline-verified', report=report)
            fresh = await self.start(address)
            expected = dict(peer_write, **planned_first)
            await self.wait_fields(fresh, expected)
            self.assertEqual(self.state.scene_reads[-1]['backend'], 'B')
            self.assertEqual((await self.close(fresh))['fields'], expected)
            self.assertEqual(self.state.attempts['UpdateLayer'], base + writers)
            self.assertEqual(self.state.attempts['CreateLayer'], 1)
            self.assertEqual(len(self.commits()), committed + 2)
            record('complete', admitted=edits, uncertain=writers, queued=edits - writers, committed=committed)

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
    Live Edit 0.1.x unverified catch-up.
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

    async def test_reopen_with_a_live_edit_0_1_unverified_catch_up_applies_own_and_peer_commits(self):
        # Live Edit 0.1.x coverage (`unverified_cut`). The reopened layer reads with the endpoint's writer ID. Without
        # a proven stable cut its snapshot is acked below the peer and own commits it holds, which are delivered
        # again, the own one included, so the layer converges on the service's state.
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
    # Each test compares the service's attempt counts across calls that must issue no RPC.
    start = PluginTests.start_answered
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
