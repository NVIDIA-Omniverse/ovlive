# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native graph attachment consumers against the independent semantic service."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import unittest

import grpc
from service import Principal, pb, rpc
from graph_service import GraphState, GraphServer
from scene_graph_wire import Command as C, Field as F, Sample as S, TimeSamples as T, Delta, Token, Specifier, Variability, Paths, PathOp, typed_value, encode, decode
from scene_graph_wire import REAL_FAMILIES, Real, String, Strings, TokenArray, real
from scene_graph_model import Model, ranked
from test_scene import RotatingProxy
import tls_fixture

AUTH = (("authorization", "Bearer synthetic-writer"),)


def typed(state):
    return [dict(node, fields={name: typed_value(value)
                              for name, value in node["fields"].items()}) for node in state]


def operands(value):
    """The native graph driver's set-value operands for one presence family (tests/native/scene/graph_value_json.h)."""
    def text(item):
        return 'x' + item.encode('utf-8').hex()
    if type(value) is String:
        return 'string ' + text(value.value)
    if type(value) in (Strings, TokenArray):
        kind = 'string[]' if type(value) is Strings else 'token[]'
        return ' '.join((kind, str(len(value.value)), *map(text, value.value)))
    width = REAL_FAMILIES[value.family][2]
    count = (str(len(value.bits) // width),) if value.array else ()
    return ' '.join((value.family + ('[]' if value.array else ''), *count, *map(str, value.bits)))


class Driver:
    def __init__(self, process, path):
        self.process, self.path = process, path
        # A driver started with its own stderr pipe keeps diagnostics in a
        # sibling artifact, so stdout carries only protocol lines.
        self.stderr_path = path.with_suffix('.stderr.log') if process.stderr is not None else None
        self.queue, self.lock, self.error = asyncio.Queue(maxsize=16), asyncio.Lock(), None
        self.drainer = asyncio.create_task(self.drain_all())

    async def drain_all(self):
        await asyncio.gather(self.drain(), *(() if self.stderr_path is None else (self.drain_stderr(),)))

    async def drain_stderr(self):
        remaining = 2 * 1024 * 1024
        with self.stderr_path.open('wb') as output:
            while line := await self.process.stderr.readline():
                output.write(line[:remaining])
                # Flush each line so log_tail sees stderr the drainer has read.
                output.flush()
                remaining = max(0, remaining - len(line))

    async def drain(self):
        remaining = 2 * 1024 * 1024
        try:
            with self.path.open('wb') as output:
                while line := await self.process.stdout.readline():
                    output.write(line[:remaining])
                    if len(line) > remaining:
                        self.error = AssertionError('native graph output exceeded artifact bound')
                    remaining = max(0, remaining - len(line))
                    if line.startswith(b'{') and self.error is None:
                        try:
                            self.queue.put_nowait(json.loads(line))
                        except (ValueError, asyncio.QueueFull) as error:
                            self.error = error
        except Exception as error:
            self.error = error
        finally:
            if not self.queue.full():
                self.queue.put_nowait(None)

    async def receive(self):
        value = await asyncio.wait_for(self.queue.get(), 12)
        if self.error:
            raise AssertionError('native graph protocol failed') from self.error
        if value is None:
            raise AssertionError(self.log_tail())
        return value

    def log_tail(self):
        text = self.path.read_text()[-8192:]
        if self.stderr_path is not None and self.stderr_path.exists():
            text += '\n--- stderr ---\n' + self.stderr_path.read_text(errors='replace')[-8192:]
        return text

    async def command(self, text):
        async with self.lock:
            self.process.stdin.write((text + '\n').encode())
            await self.process.stdin.drain()
            return await self.receive()


class GraphTransportTests(unittest.IsolatedAsyncioTestCase):
    transport_tls = False  # Proxies terminate TLS with a per-case test CA; backends stay plaintext.

    async def asyncSetUp(self):
        self.seed = int(os.environ['LIVE_TEST_SEED'])
        self.artifact = Path(os.environ['LIVE_TEST_ARTIFACT_DIR'])
        self.state = GraphState(self.seed)
        self.state.tokens = {'synthetic-writer': Principal('writer')}
        self.drivers, self.channels, self.servers, self.proxies = [], [], [], []
        self.trace = {'seed': self.seed, 'operations': []}
        self.addAsyncCleanup(self.cleanup)
        self.address, self.stub = await self.backend('A')
        self.session = (await self.stub.CreateSession(pb.CreateSessionRequest(path='graph-base', name='graph'),
                                                     metadata=AUTH, timeout=5)).session_id
        self.initial = encode(Delta(commands=(C(1, 2, 1, 'P', 6), C(1, 3, 2, 'Child', 6), C(1, 4, 3, 'value', 1),
                                               C(1, 5, 1, 'Q', 6), C(1, 6, 5, 'other', 1)),
                                    fields=(F(4, 'value', 10), F(6, 'value', 20))))
        messages = [pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name='root.live')),
                    pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=self.initial))]
        self.object = (await self.stub.CreateLayer(iter(messages), metadata=AUTH, timeout=5)).object_id

    async def backend(self, name):
        server = GraphServer(self.state, name)
        self.servers.append(server)
        address = await server.start()
        channel = grpc.aio.insecure_channel(address)
        self.channels.append(channel)
        await asyncio.wait_for(channel.channel_ready(), 5)
        return address, rpc.LiveServiceStub(channel)

    async def cleanup(self):
        failures = []
        for driver in self.drivers:
            if driver.process.returncode is None:
                try:
                    driver.process.kill()
                except ProcessLookupError:
                    pass
        for jobs in ((d.process.wait() for d in self.drivers), (d.drainer for d in self.drivers),
                     (p.close() for p in self.proxies), (c.close() for c in self.channels), (s.stop() for s in self.servers)):
            try:
                results = await asyncio.wait_for(asyncio.gather(*jobs, return_exceptions=True), 5)
                failures.extend(repr(result) for result in results if isinstance(result, BaseException))
            except Exception as error:
                failures.append(repr(error))
        self.trace.update(ledger=self.state.snapshot(), reads=self.state.reads, attempts=self.state.attempts,
                          cleanup_errors=failures,
                          transport_proxies=[p.snapshot() for p in self.proxies if hasattr(p, 'snapshot')])
        (self.artifact / (self.id().split('.')[-1] + '.graph-transport.json')).write_text(json.dumps(self.trace, indent=2) + '\n')
        self.state.assert_faults_fired()
        self.assertFalse(failures)
        self.assertTrue(all(driver.error is None for driver in self.drivers))

    async def tls_pki(self, label='ca'):
        pki = await tls_fixture.case_pki(self, label)
        self.pki = self.pkis['ca'] if 'ca' in self.pkis else None
        return pki

    async def http2_proxy(self, target):
        from http2_proxy import Http2Proxy
        proxy = Http2Proxy(target, ssl_context=(await self.tls_pki()).server_context() if self.transport_tls else None)
        self.proxies.append(proxy)
        return proxy

    async def start(self, address=None, *, host=False, environment=None):
        from tools.live.writer_campaign import admit_client
        admit_client(self)
        # Only a TLS-terminating proxy's authority needs the test CA; a driver
        # that dials a backend directly keeps plaintext.
        if environment is None and self.transport_tls and any(
                getattr(p, 'ssl_context', None) and getattr(p, 'address', None) == address for p in self.proxies):
            environment = self.pki.environment()
        process = await asyncio.create_subprocess_exec(os.environ['LIVE_TEST_SCENE_DRIVER'], '--graph-host' if host else '--graph',
                    address or self.address, self.session, 'root.live', stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=1024 * 1024,
                    env=None if environment is None else dict(os.environ, **environment))
        driver = Driver(process, self.artifact / f"{self.id().split('.')[-1]}-{len(self.drivers)}.graph-native.log")
        self.drivers.append(driver)
        state = await driver.receive()
        self.assertEqual(state['result'], 0, state)
        self.assertEqual(state['object'], self.object)
        return driver, state

    async def close(self, driver):
        result = await driver.command('close')
        self.assertEqual(result['result'], 0, result)
        await asyncio.wait_for(driver.process.wait(), 8)
        await driver.drainer
        self.assertEqual(driver.process.returncode, 0, driver.log_tail())

    def commits(self):
        return [entry for entry in self.state.ledger if entry.session == self.session and
                entry.layer == 'root.live' and entry.object_id == self.object]

    def expected(self):
        model = Model()
        for entry in self.commits():
            model.apply(entry.payload, entry.sequence, per_list=self.state.reorder_rule == 'per_list')
        return typed(model.state)

    async def update(self, delta, stub=None):
        payload = encode(delta)
        self.trace['operations'].append({'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': payload.hex()})
        messages = [pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name='root.live',
                    object_id=self.object, client_id='independent-peer')),
                    pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=payload))]
        return (await (stub or self.stub).UpdateLayer(iter(messages), metadata=AUTH, timeout=5)).sequence_num

    async def empty_update(self, stub=None):
        """Commit an upload with no data chunks: an empty delta that consumes a sequence (service SDD §8.4)."""
        messages = [pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name='root.live',
                    object_id=self.object, client_id='independent-peer'))]
        return (await (stub or self.stub).UpdateLayer(iter(messages), metadata=AUTH, timeout=5)).sequence_num

    def compare(self, state):
        self.assertEqual(state['result'], 0, state)
        self.assertEqual(state['pending'], 0, state)
        # Order values are not observable; the sibling sequence they give is.
        self.assertEqual(ranked(state['authoritative']), ranked(self.expected()))
        self.assertEqual(ranked(state['visible']), ranked(self.expected()))

    async def test_peer_structure_and_native_field_after_identity_move(self):
        first, _ = await self.start(host=True)
        second, _ = await self.start()
        sequence = await self.update(Delta(commands=(C(3, 3, 5, 'Moved', old_parent=2), C(2, 2, 1),
                                                    C(1, 100, 5, 'New', 6)), fields=(F(100, 'peer', 77),)))
        for driver in (first, second):
            self.compare(await driver.command(f'wait {sequence}'))
        submitted = await first.command('set 4 value 37')
        self.assertEqual(submitted['result'], 0, submitted)
        self.assertGreater(submitted['accepted_id'], 0)
        for driver in (first, second):
            self.compare(await driver.command(f'wait {sequence + 1}'))
        last = decode(self.commits()[-1].payload)
        self.assertEqual([(f.section, f.key, f.value) for f in last.fields], [(4, 'value', 37)])
        expected = {node['id']: node for node in self.expected()}
        self.assertEqual(expected[4]['path'], '/Q/Moved.value')
        self.assertEqual(expected[4]['fields']['value']['value'], 37)
        self.assertNotIn(2, expected)
        fresh, opened = await self.start()
        self.compare(opened)
        for driver in (first, second, fresh):
            await self.close(driver)

    async def service_limit_refusal(self, verb, expected):
        """Author `verb`, which the service refuses over its entry limits, and
        return the write's rpc_failed fields; the layer fails INVALID_ARGUMENT."""
        driver, initial = await self.start(host=True)
        base = len(self.commits())
        submitted = await driver.command(verb)
        self.assertEqual(submitted['result'], 0, submitted)
        failed = await driver.command(f'wait {base + 1}')
        self.assertEqual((failed['result'], failed['status_result'], failed['status_state'], failed['status_grpc']),
                         (1, 1, 7, 3), failed)  # INVALID_ARGUMENT, FAILED, gRPC INVALID_ARGUMENT
        self.assertEqual(len(self.commits()), base)
        self.assertEqual(self.state.rejections[-1]['kind'], expected)
        diagnostics = (await driver.command('diagnostics'))['diagnostics']
        self.trace['service_limit_diagnostics'] = diagnostics
        writes = [r for r in diagnostics['recent_records'] if r['event_name'] == 'rpc_failed' and
                  r['fields']['operation'] == 4]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]['subject_id'], initial['subject'])
        await self.close(driver)
        return writes[0]['fields']

    async def test_service_delta_too_large_refusal_reports_its_counts(self):
        # The client counts against the service's default; this service holds one entry.
        self.state.delta_entry_limit = 1
        fields = await self.service_limit_refusal('create 1 Over 5', 'delta_too_large')
        # A Create and its marker field: the appended child emits no Reorder.
        self.assertEqual({key: fields.get(key) for key in ('result', 'grpc_status', 'reason', 'struct_commands',
                                                          'set_fields', 'time_samples', 'limit', 'count')},
                         dict(result=1, grpc_status=3, reason=1, struct_commands=1, set_fields=1, time_samples=0,
                              limit=1, count=None))

    async def test_service_children_list_too_long_refusal_reports_its_counts(self):
        self.state.children_list_limit = 1
        # The root lists P and Q; reversing them is one Reorder naming both.
        fields = await self.service_limit_refusal('reverse-children 1', 'children_list_too_long')
        self.assertEqual({key: fields.get(key) for key in ('result', 'grpc_status', 'reason', 'limit', 'count',
                                                          'struct_commands')},
                         dict(result=1, grpc_status=3, reason=2, limit=1, count=2, struct_commands=None))

    def write_records(self, diagnostics, name):
        """The layer write's records of one event in a diagnostics snapshot."""
        return [r for r in diagnostics['recent_records'] if r['event_name'] == name and r['fields']['operation'] == 4]

    async def test_unavailable_refusal_that_applied_nothing_is_reissued_unchanged_after_backoff(self):
        # The service refuses the write UNAVAILABLE with its ErrorInfo twice, first with a RetryInfo of 15 ms, then
        # without one: nothing applied, so the write stays at the FIFO head and goes out again, unchanged, after
        # each backoff, and the third attempt commits it once.
        driver, initial = await self.start(host=True)
        base = len(self.commits())
        self.state.refuse_updates(1, retry_delay_ms=15)
        self.state.refuse_updates(1)
        submitted = await driver.command('set 4 value 37')
        self.assertEqual(submitted['result'], 0, submitted)
        self.compare(await driver.command(f'wait {base + 1}'))
        self.assertEqual(self.state.attempts['UpdateLayer'], 3)  # Two refusals and the commit; nothing more.
        committed = self.commits()[-1]
        self.assertEqual(len(self.commits()), base + 1)
        self.assertEqual([(f.section, f.key, f.value) for f in decode(committed.payload).fields], [(4, 'value', 37)])
        # Every attempt sent the same bytes, operation ID and writer.
        refused = self.state.refusals
        self.assertEqual(len(refused), 2)
        self.assertTrue(refused[0]['operation_id'])
        self.assertEqual({(r['operation_id'], r['client_id'], r['sha256']) for r in refused},
                         {(refused[0]['operation_id'], committed.writer, hashlib.sha256(committed.payload).hexdigest())})
        self.assertEqual([entry['outcome'] for entry in self.state.operation_log], ['APPLIED'])
        self.assertEqual([operation.operation_id for operation in self.state.operations.values()],
                         [refused[0]['operation_id']])
        # The service's retry delay is the least wait before the next attempt.
        self.assertGreaterEqual(refused[1]['monotonic_ns'] - refused[0]['monotonic_ns'], 15_000_000)
        diagnostics = (await driver.command('diagnostics'))['diagnostics']
        self.trace['refusal_diagnostics'] = diagnostics
        failed = self.write_records(diagnostics, 'rpc_failed')
        self.assertEqual([(r['fields']['result'], r['fields']['grpc_status']) for r in failed], [(8, 14)] * 2)
        retries = self.write_records(diagnostics, 'retry_scheduled')
        self.assertEqual([(r['subject_id'], r['fields']['result'], r['fields']['attempt_id'], r['fields']['reason'])
                          for r in retries], [(initial['subject'], 8, 1, 1), (initial['subject'], 8, 2, 2)])
        self.assertEqual([r['fields']['request_id'] for r in retries], [r['fields']['request_id'] for r in failed])
        # The RetryInfo delay exactly, since it exceeds every draw of attempt 1's reconnect window, below twice the
        # driver's reconnect_initial_ms of 5 ms; then attempt 2 of the reconnect schedule, below twice min(20, 5 x 2)
        # ms.
        self.assertEqual(retries[0]['fields']['delay_ms'], 15, retries[0])
        self.assertLess(retries[1]['fields']['delay_ms'], 20)
        await self.close(driver)

    async def test_unavailable_refusals_that_applied_nothing_fail_the_layer_after_the_attempt_bound(self):
        # The driver's reconnect_max_attempts is 3: the third consecutive refusal fails the layer UNAVAILABLE with
        # the edit kept. A RetryInfo delay above reconnect_maximum_ms waits that maximum, 20 ms.
        driver, _ = await self.start(host=True)
        base = len(self.commits())
        self.state.refuse_updates(3, retry_delay_ms=1000)
        submitted = await driver.command('set 4 value 37')
        self.assertEqual(submitted['result'], 0, submitted)
        failed = await driver.command(f'wait {base + 1}')
        self.assertEqual((failed['result'], failed['status_result'], failed['status_state'], failed['status_grpc']),
                         (8, 8, 7, 14), failed)  # UNAVAILABLE, FAILED, gRPC UNAVAILABLE
        self.assertEqual(failed['pending'], 1, failed)
        self.assertEqual(len(self.commits()), base)
        self.assertEqual((len(self.state.refusals), self.state.attempts['UpdateLayer']), (3, 3))
        diagnostics = (await driver.command('diagnostics'))['diagnostics']
        self.trace['refusal_diagnostics'] = diagnostics
        retries = self.write_records(diagnostics, 'retry_scheduled')
        self.assertEqual([(r['fields']['attempt_id'], r['fields']['reason'], r['fields']['delay_ms'])
                          for r in retries], [(1, 1, 20), (2, 1, 20)])
        await self.close(driver)

    async def proven_write(self, driver, value=1):
        """A confirmed write whose APPLIED reply proves the endpoint honours operation IDs."""
        base = len(self.commits())
        self.assertEqual((await driver.command(f'set 4 value {value}'))['result'], 0)
        self.compare(await driver.command(f'wait {base + 1}'))
        self.assertEqual([entry['outcome'] for entry in self.state.operation_log], ['APPLIED'])
        return base + 1

    def chain_resends(self, diagnostics):
        """The layer write's retry_scheduled records of chain re-sends (reason 3)."""
        return [r for r in self.write_records(diagnostics, 'retry_scheduled') if r['fields'].get('reason') == 3]

    async def test_unavailable_without_the_service_error_info_resolves_through_the_chain(self):
        # A plain UNAVAILABLE, one with only a RetryInfo and one whose ErrorInfo names another domain do not prove
        # that nothing applied: each attempt's outcome is unknown. The endpoint chains writes, so the write is
        # re-sent unchanged until the service answers: a commit, ALREADY_COMMITTED for an attempt that committed,
        # or a chain refusal. The third attempt commits it once, and no capacity backoff runs.
        cases = ((dict(error_info=False), dict(error_info=False)),
                 (dict(error_info=False, retry_delay_ms=5), dict(domain='other.domain')))
        for index, (first, second) in enumerate(cases):
            with self.subTest(first=first, second=second):
                self.state.operation_log.clear()
                driver, _ = await self.start(host=True)
                base = await self.proven_write(driver, 1 + index)
                updates = self.state.attempts['UpdateLayer']
                self.state.refuse_updates(1, **first)
                self.state.refuse_updates(1, **second)
                self.assertEqual((await driver.command(f'set 4 value {37 + index}'))['result'], 0)
                self.compare(await driver.command(f'wait {base + 1}'))
                self.assertEqual(len(self.commits()), base + 1)
                self.assertEqual(self.state.attempts['UpdateLayer'], updates + 3)
                self.assertEqual([entry['outcome'] for entry in self.state.operation_log], ['APPLIED', 'APPLIED'])
                diagnostics = (await driver.command('diagnostics'))['diagnostics']
                self.assertEqual([r['fields']['reason'] for r in self.write_records(diagnostics, 'retry_scheduled')],
                                 [3, 3])
                self.assertEqual([r['fields']['result'] for r in self.write_records(diagnostics, 'rpc_failed')],
                                 [19, 19])
                await self.close(driver)

    async def test_unavailable_refusal_of_a_resend_backs_off_and_resolves_the_lost_reply(self):
        # The first attempt commits and its reply is lost; the service refuses the re-send with its ErrorInfo.
        # The refusal applied nothing, so the write is re-sent after the backoff, and the service answers that
        # re-send ALREADY_COMMITTED with the first attempt's position: the write committed once.
        driver, _ = await self.start(host=True)
        base = await self.proven_write(driver)
        updates = self.state.attempts['UpdateLayer']
        lost = self.state.plan('UpdateLayer', 'after_commit', pause=True, code=grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await driver.command('set 4 value 37'))['result'], 0)
        await asyncio.wait_for(lost.reached.wait(), 5)
        self.state.refuse_updates(1, retry_delay_ms=5)
        lost.release.set()
        self.compare(await driver.command(f'wait {base + 1}'))
        self.assertEqual(len(self.commits()), base + 1)  # The first attempt committed once.
        self.assertEqual(self.state.attempts['UpdateLayer'], updates + 3)
        self.assertEqual(len(self.state.refusals), 1)
        self.assertEqual([entry['outcome'] for entry in self.state.operation_log],
                         ['APPLIED', 'APPLIED', 'ALREADY_COMMITTED'])
        diagnostics = (await driver.command('diagnostics'))['diagnostics']
        self.assertEqual([r['fields']['reason'] for r in self.write_records(diagnostics, 'retry_scheduled')], [3, 1])
        await self.close(driver)

    async def test_a_plain_definitive_status_on_a_chained_write_is_settled_through_the_chain(self):
        # Only OK, an error with the service's ErrorInfo or INVALID_ARGUMENT without one settles a chained call. A
        # plain RESOURCE_EXHAUSTED or PERMISSION_DENIED may follow a commit, so the write's outcome is unknown: it is
        # re-sent with no_wait, which the service answers ALREADY_COMMITTED for an attempt that committed, or commits.
        cases = (('after_commit', grpc.StatusCode.RESOURCE_EXHAUSTED, 'ALREADY_COMMITTED'),
                 ('before', grpc.StatusCode.PERMISSION_DENIED, None))
        for index, (boundary, code, outcome) in enumerate(cases):
            with self.subTest(boundary=boundary, code=code):
                self.state.operation_log.clear()
                driver, _ = await self.start(host=True)
                base = await self.proven_write(driver, 1 + index)
                updates = self.state.attempts['UpdateLayer']
                self.state.plan('UpdateLayer', boundary, code=code)
                self.assertEqual((await driver.command(f'set 4 value {37 + index}'))['result'], 0)
                self.compare(await driver.command(f'wait {base + 1}'))
                self.assertEqual(len(self.commits()), base + 1)  # Committed once.
                self.assertEqual(self.state.attempts['UpdateLayer'], updates + 2)
                self.assertEqual([entry['outcome'] for entry in self.state.operation_log],
                                 ['APPLIED', 'APPLIED'] + ([outcome] if outcome else []))
                diagnostics = (await driver.command('diagnostics'))['diagnostics']
                self.assertEqual([r['fields']['reason'] for r in self.write_records(diagnostics, 'retry_scheduled')],
                                 [3])
                await self.close(driver)

    async def test_a_lost_reply_settles_before_the_read_resumes_into_a_net_empty_catch_up(self):
        # A chained write commits and its reply is held; the service then ends the read with INTERNAL, and the layer
        # observes that before the reply is lost. The read stays closed until the write's no_wait re-send is answered
        # ALREADY_COMMITTED with its position, so it resumes from that position. Meanwhile a peer committed two
        # uploads without content, so the resumed read's catch-up nets to nothing and the service sends only the
        # caught-up header above the cursor.
        from service import Fault
        driver, _ = await self.start(host=True)
        base = await self.proven_write(driver)
        lost = self.state.plan('UpdateLayer', 'after_commit', pause=True, code=grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await driver.command('set 4 value 37'))['result'], 0)
        await asyncio.wait_for(lost.reached.wait(), 5)
        generation = (await driver.command('report'))['generation']
        reads = len(self.state.reads)
        attempt = self.state.reads[-1]['attempt']
        self.state.faults.append(Fault('SubscribeReadLayer', attempt, f'before_frame:{base + 2}',
                                       grpc.StatusCode.INTERNAL, False))
        self.assertEqual(await self.empty_update(), base + 2)
        deadline = asyncio.get_running_loop().time() + 10  # Failure-only watchdog.
        while (await driver.command('report'))['generation'] == generation:
            self.assertLess(asyncio.get_running_loop().time(), deadline, 'the layer never observed the read end')
        final = await self.empty_update()
        lost.release.set()
        self.compare(await driver.command(f'wait {final}'))
        self.assertEqual(len(self.commits()), final)  # The write committed once, at base + 1.
        self.assertEqual([entry['outcome'] for entry in self.state.operation_log],
                         ['APPLIED', 'APPLIED', 'ALREADY_COMMITTED'])
        self.assertEqual([read['cursor'] for read in self.state.reads[reads:]], [base + 1])
        self.assertEqual([(entry['cursor'], entry['latest']) for entry in self.state.net_empty_catch_ups],
                         [(base + 1, final)])
        await self.close(driver)

    async def test_a_chain_proven_write_is_delivered_by_a_resubscribed_read(self):
        # With the read open, A commits and its reply is held; B, chained after it, commits and is acknowledged; A's
        # reply is then lost. B's acknowledgment proves A committed, but the open read filtered A's commit and the
        # service reports no position for it, so the layer resubscribes once from its coverage frontier, below A.
        # The graph reference's catch-up coalesces A's and B's commits into one aggregate, which retires both
        # intents. Neither write is re-sent. (Its per-row versions keep a diff of overwritten fields nonempty, so
        # a net-empty catch-up over this writer's writes is covered by the progress matrix and the journal tests.)
        driver, _ = await self.start(host=True)
        base = await self.proven_write(driver)
        lost = self.state.plan('UpdateLayer', 'after_commit', pause=True, code=grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await driver.command('set 4 value 37'))['result'], 0)
        await asyncio.wait_for(lost.reached.wait(), 5)
        reads = len(self.state.reads)
        submitted = await driver.command('set 4 value 38')
        self.assertEqual(submitted['result'], 0, submitted)
        acknowledged = await driver.command(f"acknowledged {submitted['accepted_id']}")
        self.assertEqual(acknowledged['result'], 0, acknowledged)
        lost.release.set()
        self.compare(await driver.command(f'wait {base + 2}'))
        self.assertEqual(len(self.commits()), base + 2)
        self.assertEqual([entry['outcome'] for entry in self.state.operation_log], ['APPLIED', 'APPLIED', 'APPLIED'])
        self.assertEqual([read['cursor'] for read in self.state.reads[reads:]], [base])
        await self.close(driver)

    async def test_a_fresh_reader_opens_a_layer_whose_snapshot_keeps_erased_fields(self):
        # The service's compacted snapshot keeps an erased field as a present but empty value.
        sequence = await self.update(Delta(fields=(F(4, 'value', None), F(6, 'value', None))))
        layer = self.state.sessions[self.session].layers['root.live']
        self.assertEqual(layer.erased, frozenset({(4, 'value'), (6, 'value')}))
        fresh, opened = await self.start()
        self.compare(opened)
        self.assertGreaterEqual(opened['covered'], sequence)
        expected = {node['id']: node for node in self.expected()}
        self.assertNotIn('value', expected[4]['fields'])
        await self.close(fresh)

    async def test_a_fresh_reader_applies_commits_its_snapshot_already_holds(self):
        # Without a proven stable cut the service acks a fresh snapshot at its pin and delivers again the commits
        # the snapshot already holds, including a Create of a held section and a Reorder naming a moved child.
        await self.update(Delta(commands=(C(1, 100, 1, 'R', 6), C(3, 5, 100, 'Q', old_parent=1)),
                                fields=(F(100, 'peer', 1),)))
        await self.update(Delta(commands=(C(4, 1, children=(2, 100)),), fields=(F(100, 'peer', 2),)))
        last = await self.update(Delta(commands=(C(3, 2, 100, 'P', old_parent=1),)))
        self.state.unverified_cut = 3
        fresh, _ = await self.start()
        self.compare(await fresh.command(f'wait {last}'))
        expected = {node['id']: node for node in self.expected()}
        self.assertEqual(expected[3]['path'], '/R/P/Child')
        self.assertEqual(expected[100]['fields']['peer']['value'], 2)
        await self.close(fresh)

    async def test_a_reader_applies_updates_after_empty_commits(self):
        # An empty commit consumes a sequence. The service delivers it as an empty diff based at the previous
        # sequence; the reader also accepts a header-only frame or nothing for it, and applies the updates after it.
        for value, form in enumerate(("diff", "header", "silent")):
            with self.subTest(form=form):
                self.state.empty_commit_form = form
                driver, opened = await self.start()
                self.compare(opened)
                empty = await self.empty_update()
                final = await self.update(Delta(fields=(F(4, 'value', 50 + value),)))
                self.assertEqual(final, empty + 1)
                self.compare(await driver.command(f'wait {final}'))
                await self.close(driver)

    async def test_a_resume_across_only_empty_commits_catches_up(self):
        # A catch-up that owes nothing sends only the caught-up header above the reader's cursor.
        other_address, other_stub = await self.backend('B')
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        driver, opened = await self.start(await proxy.start())
        self.compare(opened)
        proxy.rotate(other_address, pause=True)
        await self.empty_update(other_stub)
        final = await self.empty_update(other_stub)
        proxy.gate.set()
        caught = await driver.command(f'wait {final}')
        self.compare(caught)
        self.assertGreater(caught['generation'], opened['generation'])
        self.assertTrue(any(read['backend'] == 'B' and read['cursor'] == opened['covered'] for read in self.state.reads))
        later = await self.update(Delta(fields=(F(4, 'value', 52),)), other_stub)
        self.compare(await driver.command(f'wait {later}'))
        await self.close(driver)

    async def test_a_read_the_service_ends_with_internal_resubscribes(self):
        # The service ends a read with INTERNAL during a backend incident. A read is not a mutation, so the layer
        # resubscribes from its coverage.
        from service import Fault
        driver, opened = await self.start()
        self.compare(opened)
        attempt = self.state.reads[-1]['attempt']
        self.state.faults.append(Fault('SubscribeReadLayer', attempt, f"before_frame:{opened['covered'] + 1}",
                                       grpc.StatusCode.INTERNAL, False))
        final = await self.update(Delta(fields=(F(4, 'value', 61),)))
        self.compare(await driver.command(f'wait {final}'))
        self.assertEqual(self.state.attempts['SubscribeReadLayer'], attempt + 1)
        await self.close(driver)

    async def test_backend_rotation_reconstructs_graph_diff_and_fresh_observer(self):
        other_address, other_stub = await self.backend('B')
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        address = await proxy.start()
        driver, opened = await self.start(address)
        self.compare(opened)
        self.assertGreater(opened['covered'], 0)
        self.assertGreater(opened['applied'], 0)
        proxy.rotate(other_address, pause=True)
        await self.update(Delta(commands=(C(3, 3, 5, 'Moved', old_parent=2),)), other_stub)
        await self.update(Delta(commands=(C(2, 2, 1),)), other_stub)
        final = await self.update(Delta(commands=(C(1, 100, 5, 'New', 6),), fields=(F(4, 'value', 81),)), other_stub)
        proxy.gate.set()
        caught = await driver.command(f'wait {final}')
        self.compare(caught)
        self.assertGreater(caught['generation'], opened['generation'])
        self.assertTrue(any(read['backend'] == 'B' and read['cursor'] == opened['covered'] for read in self.state.reads))
        fresh, ready = await self.start(other_address)
        self.compare(ready)
        await self.close(driver)
        await self.close(fresh)

    async def test_a_resume_diff_places_a_new_sibling_among_unchanged_ones(self):
        # The service restates only changed sections, each at its index in its parent's current children list, so
        # an unchanged sibling the resumer holds can carry the index a new sibling is restated at.
        other_address, other_stub = await self.backend('B')
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        driver, opened = await self.start(await proxy.start())
        self.compare(opened)
        proxy.rotate(other_address, pause=True)
        final = await self.update(Delta(commands=(C(2, 2, 1), C(1, 100, 1, 'R', 6))), other_stub)
        proxy.gate.set()
        self.compare(await driver.command(f'wait {final}'))
        expected = {node['id']: node for node in self.expected()}
        self.assertEqual(expected[1]['fields']['primChildren']['value'], ['Q', 'R'])
        await self.close(driver)

    async def test_resume_diff_retires_the_acknowledged_local_write_it_covers(self):
        other_address, _ = await self.backend('B')
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        driver, opened = await self.start(await proxy.start(), host=True)
        base = opened['covered']
        hold = self.state.plan_read_frame(self.state.reads[-1]['attempt'], base + 1)
        await self.update(Delta(fields=(F(6, 'value', 21),)))  # Peer at base+1, held before its frame.
        await asyncio.wait_for(hold.reached.wait(), 5)
        submitted = await driver.command('set 4 value 37')  # Own write commits at base+2.
        self.assertEqual(submitted['result'], 0, submitted)
        acknowledged = await driver.command(f"acknowledged {submitted['accepted_id']}")
        self.assertEqual(acknowledged['result'], 0, acknowledged)
        await proxy.rotate_and_wait(other_address)
        hold.release.set()
        caught = await driver.command(f'wait {base + 2}')
        self.compare(caught)
        self.assertTrue(any(read['backend'] == 'B' and read['cursor'] == base for read in self.state.reads))
        self.assertEqual(self.state.attempts['UpdateLayer'], 2)  # The peer's and ours; nothing reissued.
        await self.close(driver)

    async def test_native_sample_writers_preserve_keys_replace_maps_and_resume(self):
        def bits(time):
            return struct.unpack('<Q', struct.pack('<d', time))[0]
        def sample_field(state):
            return next(n for n in state['visible'] if n['id'] == 4)['fields']['timeSamples']
        other_address, other_stub = await self.backend('B')
        proxy = RotatingProxy(self.address); self.proxies.append(proxy)
        first, opened = await self.start(await proxy.start(), host=True)
        second, _ = await self.start(host=True)
        base = await self.update(Delta(fields=(F(4, 'timeSamples', T(((-0.0, 1), (2.0, 2)))),)))
        for driver in (first, second):
            self.compare(await driver.command(f'wait {base}'))
        originals = []
        fault = self.state.plan('UpdateLayer', 'before', pause=True)
        try:
            own = await first.command(f'sample 4 {bits(1.0)} 17')
            self.assertEqual(own['result'], 0, own)
            await asyncio.wait_for(fault.reached.wait(), 5)
            original = own['pending_intents'][0]['bytes']; originals.append(original)
            self.assertEqual(decode(bytes.fromhex(original)).samples, (S(4, 1.0, 17),))
            self.assertFalse(decode(bytes.fromhex(original)).fields)
            peer = await second.command(f'sample 4 {bits(2.0)} 29')
            self.assertEqual(peer['result'], 0, peer)
            await second.command(f'wait {base + 1}')
            pending = await first.command(f'journal-applied {base + 1}')
            self.assertEqual(pending['result'], 0, pending)
            self.assertGreaterEqual(pending['journal_applied'], base + 1)
            self.assertEqual([p['bytes'] for p in pending['pending_intents']], [original])
            self.assertEqual(sample_field(pending), typed_value(T(((-0.0, 1), (1.0, 17), (2.0, 29)))))
        finally:
            fault.release.set()
        for driver in (first, second):
            self.compare(await driver.command(f'wait {base + 2}'))
        self.assertEqual(self.commits()[-1].payload.hex(), original)
        # Whole-map authoring compiles to sample records: it erases the keys it
        # saw and sets its own, so a concurrent peer key at another time survives.
        # Both original intents retain bytes.
        fault = self.state.plan('UpdateLayer', 'before', pause=True)
        try:
            replacement = await first.command(f'sample-map 4 {bits(-0.0)} 42')
            self.assertEqual(replacement['result'], 0, replacement)
            await asyncio.wait_for(fault.reached.wait(), 5)
            original = replacement['pending_intents'][0]['bytes']; originals.append(original)
            self.assertFalse(decode(bytes.fromhex(original)).fields)
            self.assertEqual(decode(bytes.fromhex(original)).samples, (S(4, 1.0, None), S(4, 2.0, None), S(4, -0.0, 42)))
            peer = await second.command(f'sample 4 {bits(3.0)} 31')
            self.assertEqual(peer['result'], 0, peer)
            await second.command(f'wait {base + 3}')
            pending = await first.command(f'journal-applied {base + 3}')
            self.assertEqual(pending['result'], 0, pending)
            self.assertGreaterEqual(pending['journal_applied'], base + 3)
            self.assertEqual(sample_field(pending), typed_value(T(((-0.0, 42), (3.0, 31)))))
            self.assertEqual([p['bytes'] for p in pending['pending_intents']], [original])
        finally:
            fault.release.set()
        for driver in (first, second):
            self.compare(await driver.command(f'wait {base + 4}'))
        self.assertEqual(self.commits()[-1].payload.hex(), original)
        await proxy.rotate_and_wait(other_address, pause=True)
        # Resume must carry a key-sign-only map change and the independent key
        # edit after it; source scene and fresh observers must agree exactly.
        await self.update(Delta(fields=(F(4, 'timeSamples', T(((0.0, 42),))),)), other_stub)
        final = await self.update(Delta(samples=(S(4, 4.0, False),)), other_stub)
        proxy.gate.set()
        caught = await first.command(f'wait {final}')
        self.compare(caught)
        self.compare(await second.command(f'wait {final}'))
        self.assertEqual(sample_field(caught), typed_value(T(((0.0, 42), (4.0, False)))))
        self.assertGreater(caught['generation'], opened['generation'])
        self.assertTrue(any(r['backend']=='B' and r['cursor']==base+4 for r in self.state.reads))
        fresh, state = await self.start(other_address); self.compare(state)
        self.assertEqual(sample_field(state), sample_field(caught))
        for payload in originals:
            self.assertEqual(sum(c.payload.hex()==payload for c in self.commits()), 1)
        self.assertEqual(self.state.attempts['UpdateLayer'], 7)
        self.trace['original_sample_intents'] = originals
        for driver in (first, second, fresh):
            await self.close(driver)

    async def test_presence_value_families_converge_through_writers_resume_and_reopen(self):
        # The users.live shape Kit's presence layer authors: string bound_camera and following_user, a string-array
        # selection, and a camera with a matrix4d pose, a token-array op order and float, float2, float4[], double3
        # and double attributes. The independent peer and the production writer each author every family, and the
        # writer's committed bytes decode in the independent profile to exactly the values it was given.
        def matrix(*overrides):
            components = [float(i == j) for i in range(4) for j in range(4)]
            for index, component in overrides:
                components[index] = component
            return real('matrix4d', *components)

        other_address, other_stub = await self.backend('B')
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        writer, _ = await self.start(host=True)
        observer, opened = await self.start(await proxy.start())
        attributes = {102: ('bound_camera', 'string', String('perspective')),
                      103: ('following_user', 'string', String('')),
                      104: ('selected_prim_paths', 'string[]', Strings(('/World/Cube',))),
                      106: ('xformOp:transform', 'matrix4d', matrix((12, 5.0))),
                      107: ('xformOpOrder', 'token[]', TokenArray(('xformOp:transform',))),
                      108: ('focalLength', 'float', real('float', 50.0)),
                      109: ('clippingRange', 'float2', real('vec2f', 1.0, 1000000.0)),
                      110: ('clippingPlanes', 'float4[]', Real('vec4f', (), True)),
                      111: ('omni:kit:centerOfInterest', 'double3', real('vec3d', 0.0, 0.0, -500.0)),
                      112: ('shutter:open', 'double', 0.0),
                      113: ('exposure', 'float', real('float', 0.0))}
        commands = [C(1, 100, 1, '__session_shared_data__', 6), C(1, 101, 100, '_user', 6),
                    C(1, 105, 101, 'perspective', 6)]
        fields = [F(101, 'specifier', Specifier(0)), F(105, 'specifier', Specifier(0)), F(105, 'typeName', Token('Camera')),
                  F(101, 'floats', real('float', 1.5, -0.0, array=True)),
                  F(101, 'doubles', real('double', 2.5, array=True)),
                  F(101, 'transforms', Real('matrix4d', matrix().bits * 2, True)),
                  F(101, 'ranges', real('vec2f', 0.25, 4.0, array=True)),
                  F(101, 'pivots', real('vec3d', 1.0, 2.0, 3.0, array=True)),
                  F(101, 'plane', real('vec4f', 0.0, 1.0, 0.0, 0.0))]
        for section, (name, type_name, value) in attributes.items():
            commands.append(C(1, section, 105 if section > 105 else 101, name, 1))
            fields += [F(section, 'typeName', Token(type_name)), F(section, 'default', value)]
        fields += [F(section, 'custom', True) for section in (102, 103, 104)] + [F(107, 'variability', Variability(1))]
        base = await self.update(Delta(commands=tuple(commands), fields=tuple(fields)))
        for driver in (writer, observer):
            self.compare(await driver.command(f'wait {base}'))

        # The writer's edits: following, selection, a camera move and lens change, including a matrix component and
        # a focal length that only change sign and a NaN payload, then every scalar and array counterpart.
        nan_payload = 0x7fc00123
        edits = ((103, 'default', String('user-λ')), (102, 'default', String('')),
                 (104, 'default', Strings(('/World/Cube', '/World/Sphere', ''))),
                 (106, 'default', matrix((12, 5.0), (13, -0.0), (14, -2.5))), (108, 'default', real('float', 0.0)),
                 (108, 'default', real('float', -0.0)), (113, 'default', Real('float', (nan_payload,))),
                 (109, 'default', real('vec2f', 0.5, 2000.0)),
                 (110, 'default', real('vec4f', 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 1.0, -0.0, array=True)),
                 (111, 'default', real('vec3d', 0.0, -0.0, -250.0)),
                 (107, 'default', TokenArray(('xformOp:translate', 'xformOp:transform'))),
                 (101, 'floats', Real('float', (nan_payload, 0), True)),
                 (101, 'doubles', real('double', -0.0, 1e300, array=True)),
                 (101, 'transforms', Real('matrix4d', matrix((15, -0.0)).bits, True)),
                 (101, 'ranges', real('vec2f', -0.0, 1.0, array=True)),
                 (101, 'pivots', Real('vec3d', (), True)),
                 (101, 'plane', real('vec4f', 0.0, 0.0, 1.0, -0.0)))
        for section, field, value in edits:
            submitted = await writer.command(f'set-value {section} {field} {operands(value)}')
            self.assertEqual(submitted['result'], 0, submitted)
        authored = base + len(edits)
        for driver in (writer, observer):
            self.compare(await driver.command(f'wait {authored}'))
        written = [decode(entry.payload) for entry in self.commits()[-len(edits):]]
        self.assertEqual([(f.section, f.key, f.value) for delta in written for f in delta.fields], list(edits))

        # Removals: the writer clears the selection, the peer clears following_user and deletes clippingPlanes.
        cleared = await writer.command('erase-field 104 default')
        self.assertEqual(cleared['result'], 0, cleared)
        self.compare(await writer.command(f'wait {authored + 1}'))
        self.assertEqual([(f.section, f.key, f.value) for f in decode(self.commits()[-1].payload).fields],
                         [(104, 'default', None)])
        removed = await self.update(Delta(commands=(C(2, 110, 105),), fields=(F(103, 'default', None),)))
        for driver in (writer, observer):
            self.compare(await driver.command(f'wait {removed}'))
        expected = {node['id']: node for node in self.expected()}
        self.assertNotIn('default', expected[104]['fields'])
        self.assertNotIn('default', expected[103]['fields'])
        self.assertNotIn(110, expected)

        # Resume: on the second backend the peer changes only signs and a NaN payload, and erases an array; the
        # observer's catch-up diff must carry each representation change.
        covered = (await observer.command('report'))['covered']
        proxy.rotate(other_address, pause=True)
        await self.update(Delta(fields=(F(106, 'default', matrix((12, 5.0), (13, 0.0), (14, -2.5))),
                                        F(108, 'default', real('float', 0.0)))), other_stub)
        final = await self.update(Delta(fields=(F(113, 'default', Real('float', (nan_payload + 1,))),
                                                F(101, 'pivots', None), F(102, 'default', String('/World/Camera')))),
                                  other_stub)
        proxy.gate.set()
        caught = await observer.command(f'wait {final}')
        self.compare(caught)
        self.compare(await writer.command(f'wait {final}'))
        self.assertGreater(caught['generation'], opened['generation'])
        self.assertTrue(any(read['backend'] == 'B' and read['cursor'] == covered for read in self.state.reads))

        # Reopen: a fresh subscriber's snapshot holds the same exact representations.
        fresh, reopened = await self.start(other_address)
        self.compare(reopened)
        nodes = {node['id']: node['fields'] for node in reopened['visible']}
        self.assertEqual(nodes[106]['default'], typed_value(matrix((12, 5.0), (13, 0.0), (14, -2.5))))
        self.assertEqual(nodes[108]['default'], typed_value(real('float', 0.0)))
        self.assertEqual(nodes[113]['default'], typed_value(Real('float', (nan_payload + 1,))))
        self.assertEqual(nodes[102]['default'], typed_value(String('/World/Camera')))
        self.assertEqual(nodes[107]['default'], typed_value(edits[10][2]))
        self.assertNotIn('pivots', nodes[101])
        self.assertEqual(self.state.attempts['UpdateLayer'], len(edits) + 5)  # Nothing reissued.
        for driver in (writer, observer, fresh):
            await self.close(driver)

    async def test_service_execution_uses_latest_subtree_with_peer_insertions(self):
        driver, opened = await self.start()
        self.compare(opened)
        operations = (
            (C(3, 3, 5, 'Moved', old_parent=2), C(1, 100, 3, 'PeerChild', 6)),
            (C(2, 3, 5), C(1, 101, 100, 'Grandchild', 6)),
        )
        for structural, inserted in operations:
            fault = self.state.plan('UpdateLayer', 'before', pause=True)
            delayed = asyncio.create_task(self.update(Delta(commands=(structural,))))
            try:
                await asyncio.wait_for(fault.reached.wait(), 5)
                peer_commands = (inserted, C(3, 4, 5, 'value', old_parent=3)) if structural.tag == 3 else (inserted,)
                peer_sequence = await self.update(Delta(commands=peer_commands, fields=(F(inserted.section, 'peer', 77),)))
                self.compare(await driver.command(f'wait {peer_sequence}'))
            finally:
                fault.release.set()
                sequence = await asyncio.wait_for(delayed, 5)
            caught = await driver.command(f'wait {sequence}')
            self.compare(caught)
            nodes = {node['id']: node for node in caught['authoritative']}
            self.assertEqual(nodes[4]['path'], '/Q.value')
            self.assertEqual(nodes[4]['fields']['value']['value'], 10)
            if structural.tag == 3:
                self.assertEqual(nodes[100]['path'], '/Q/Moved/PeerChild')
                self.assertEqual(nodes[100]['fields']['peer']['value'], 77)
            else:
                self.assertTrue({3, 100, 101}.isdisjoint(nodes))
            fresh, ready = await self.start()
            self.compare(ready)
            await self.close(fresh)
        self.assertEqual(len(self.commits()), 5)
        self.assertEqual(self.state.attempts['UpdateLayer'], 4)
        await self.close(driver)

    async def test_seeded_concurrent_field_writers_on_nonroot_identities(self):
        seeds = [int.from_bytes(hashlib.sha256(f'graph-writer:{self.seed}:{index}'.encode()).digest()[:8], 'big') for index in range(3)]
        plans = []
        for index, seed in enumerate(seeds):
            rng = random.Random(seed)
            plans.append([(rng.choice((4, 6)), 'shared' if step % 3 == 0 else f'writer{index}',
                           rng.randint(-10000, 10000)) for step in range(12)])
        plan = {'seed': self.seed, 'writer_seeds': seeds, 'plans': plans}
        (self.artifact / (self.id().split('.')[-1] + '.plans.json')).write_text(json.dumps(plan, indent=2) + '\n')
        self.trace['writer_plan'] = plan
        drivers = [await self.start(host=True) for _ in range(3)]
        async def write(index, driver):
            for section, key, value in plans[index]:
                result = await driver.command(f'set {section} {key} {value}')
                self.assertEqual(result['result'], 0, result)
                self.assertGreater(result['accepted_id'], 0)
        await asyncio.gather(*(write(index, driver) for index, (driver, _) in enumerate(drivers)))
        final = 1 + sum(map(len, plans))
        for driver, _ in drivers:
            self.compare(await driver.command(f'wait {final}'))
        self.assertEqual(len(self.commits()), final)
        self.assertEqual(self.state.attempts['UpdateLayer'], final - 1)
        planned = sorted(item for plan in plans for item in plan)
        committed = sorted((field.section, field.key, field.value) for entry in self.commits()[1:] for field in decode(entry.payload).fields)
        self.assertEqual(committed, planned)
        # Every admission appears once, including overwritten same-field edits.
        self.assertTrue(all(not decode(entry.payload).commands for entry in self.commits()[1:]))
        fresh, ready = await self.start()
        self.compare(ready)
        for driver in [item[0] for item in drivers] + [fresh]:
            await self.close(driver)

    async def test_lost_ack_retains_nonroot_intent_without_resend(self):
        await self.lost_ack('set 4 value 99')

    async def test_lost_ack_retains_captured_subtree_move_without_resend(self):
        await self.lost_ack('move 2 /Moved')

    async def lost_ack(self, command):
        self.state.advertise_capabilities = False  # The unknown outcome is kept, never re-sent.
        driver, initial = await self.start(host=True)
        fault = self.state.plan('UpdateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        submitted = await driver.command(command)
        self.assertEqual(submitted['result'], 0)
        self.assertGreater(submitted['accepted_id'], 0)
        await asyncio.wait_for(fault.reached.wait(), 5)
        failed = await driver.command('wait 2')
        self.assertEqual(failed['result'], 19, failed)
        self.assertEqual(failed['status_result'], 19)
        self.assertGreater(failed['incident'], 0)
        self.assertEqual(failed['subject'], initial['subject'])
        self.assertEqual(failed['pending'], 1)
        self.assertEqual(failed['authoritative'], initial['authoritative'])
        self.assertEqual(failed['visible'], self.expected())
        self.assertEqual(len(self.commits()), 2)
        self.assertEqual(failed['pending_intents'], [dict(id=submitted['accepted_id'], bytes=self.commits()[-1].payload.hex())])
        self.assertEqual(self.state.attempts['UpdateLayer'], 1)
        fresh, ready = await self.start()
        self.compare(ready)
        self.assertEqual(self.state.attempts['UpdateLayer'], 1)
        await self.close(driver)
        await self.close(fresh)

    async def test_captured_namespace_edits_include_latest_peer_subtree(self):
        driver, initial = await self.start(host=True)
        self.compare(initial)
        observer, _ = await self.start()
        for move in (True, False):
            with self.subTest(move=move):
                fault = self.state.plan('UpdateLayer', 'before', pause=True)
                submitted = await driver.command('move 2 /Moved' if move else 'erase 2')
                try:
                    self.assertEqual(submitted['result'], 0, submitted)
                    self.assertGreater(submitted['accepted_id'], 0)
                    await asyncio.wait_for(fault.reached.wait(), 5)
                    commands = (C(1, 100, 2, 'Peer', 6),) if move else (
                        C(1, 101, 100, 'Grandchild', 6), C(3, 4, 5, 'escaped', old_parent=3))
                    peer = await self.update(Delta(commands=commands, fields=(F(100 if move else 101, 'peer', 77),)))
                finally:
                    fault.release.set()
                for client in (driver, observer):
                    self.compare(await client.command(f'wait {peer + 1}'))
                nodes = {node['id']: node for node in self.expected()}
                if move:
                    self.assertEqual(nodes[100]['path'], '/Moved/Peer')
                else:
                    self.assertTrue({2, 3, 100, 101}.isdisjoint(nodes))
                    self.assertEqual(nodes[4]['path'], '/Q.escaped')
                    self.assertEqual(nodes[4]['fields']['value']['value'], 10)
                own = decode(self.commits()[-1].payload)
                self.assertFalse(any(command.tag == 1 for command in own.commands))
                recursive = [command for command in own.commands if command.tag in (2, 3)]
                self.assertEqual([(command.tag, command.section) for command in recursive], [(3 if move else 2, 2)])
                fresh, ready = await self.start()
                self.compare(ready)
                await self.close(fresh)
        self.assertEqual(len(self.commits()), 5)
        self.assertEqual(self.state.attempts['UpdateLayer'], 4)
        await self.close(driver)
        await self.close(observer)

    async def test_captured_creates_keep_identity_across_writers_parent_move_and_rotation(self):
        rng = random.Random(self.seed)
        markers = [rng.randrange(1, 1000000) for _ in range(6)]
        self.trace['create_plan'] = dict(seed=self.seed, markers=markers, writers=3)
        # Persist authored choices before any native writer runs. Service order
        # and allocated IDs are observations, not a claimed reproducible schedule.
        (self.artifact / (self.id().split('.')[-1] + '.plan.json')).write_text(
            json.dumps(self.trace['create_plan'], indent=2) + '\n')
        other_address, other_stub = await self.backend('B')
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        address = await proxy.start()
        drivers = [await self.start(address, host=True) for _ in range(3)]
        # Each writer's first write is held at its commit step; its chained second write waits behind it.
        faults = [self.state.plan_write('hold', position=1, writer=initial['writer']) for _, initial in drivers]
        accepted, identities = [], []
        try:
            for index, (driver, _) in enumerate(drivers):
                first = await driver.command(f'create 2 Writer{index} {markers[index * 2]}')
                self.assertEqual(first['result'], 0, first)
                await asyncio.wait_for(faults[index].reached.wait(), 5)
                first_wire = decode(bytes.fromhex(first['pending_intents'][0]['bytes']))
                self.assertEqual(len(first_wire.commands), 1)
                command = first_wire.commands[0]
                self.assertEqual((command.tag, command.parent, command.name), (1, 2, f'Writer{index}'))
                identities.append(command.section)
                second = await driver.command(f'create {command.section} Child {markers[index * 2 + 1]}')
                self.assertEqual(second['result'], 0, second)
                self.assertEqual(second['pending'], 2)
                self.assertNotEqual(second['accepted_id'], first['accepted_id'])
                self.assertEqual(second['pending_intents'][0], first['pending_intents'][0])
                accepted.extend(second['pending_intents'])
            self.assertEqual(len(set(identities)), 3)
            # Move the parent and reorder old/new siblings before any of the
            # six accepted Creates can execute. The authored parent IDs survive.
            peer = await self.update(Delta(commands=(C(3, 2, 1, 'Moved', old_parent=1),
                           C(1, 100, 2, 'Peer', 6), C(4, 2, children=(100, 3)))))
        finally:
            for fault in faults:
                fault.release.set()
        final = peer + 6
        for driver, _ in drivers:
            self.compare(await driver.command(f'wait {final}'))
        own = [commit for commit in self.commits() if commit.writer != 'independent-peer' and commit.operation == 'UpdateLayer']
        self.assertEqual(sorted(commit.payload.hex() for commit in own), sorted(item['bytes'] for item in accepted))
        self.assertEqual(len({commit.writer for commit in own}), 3)
        nodes = {node['id']: node for node in self.expected()}
        for index, identity in enumerate(identities):
            self.assertEqual(nodes[identity]['path'], f'/Moved/Writer{index}')
            self.assertEqual(nodes[identity]['fields']['marker']['value'], markers[index * 2])
            child = next(node for node in nodes.values() if node['parent'] == identity)
            self.assertEqual(child['path'], f'/Moved/Writer{index}/Child')
            self.assertEqual(child['fields']['marker']['value'], markers[index * 2 + 1])
        self.assertEqual(nodes[2]['fields']['primChildren']['value'][:2], ['Peer', 'Child'])
        # Rotate only after confirmed work. A diff includes a new peer identity
        # while omitting another created/deleted identity; the service retains it.
        await proxy.rotate_and_wait(other_address, pause=True)
        await self.update(Delta(commands=(C(1, 101, 2, 'Transient', 6), C(2, 101, 2))), other_stub)
        final = await self.update(Delta(commands=(C(1, 102, 2, 'AfterCut', 6),)), other_stub)
        proxy.gate.set()
        for driver, opened in drivers:
            caught = await driver.command(f'wait {final}')
            self.compare(caught)
            self.assertGreater(caught['generation'], opened['generation'])
        resumed = await drivers[0][0].command('create 2 Resumed 77')
        self.assertEqual(resumed['result'], 0, resumed)
        for driver, _ in drivers:
            self.compare(await driver.command(f'wait {final + 1}'))
        fresh, opened = await self.start(other_address)
        self.compare(opened)
        self.assertEqual(self.state.attempts['CreateLayer'], 1)
        self.assertEqual(self.state.attempts['UpdateLayer'], 10)
        for driver, _ in drivers:
            await self.close(driver)
        await self.close(fresh)

    async def test_captured_create_reusing_a_retired_id_commits(self):
        driver, opened = await self.start(host=True)
        fault = self.state.plan('UpdateLayer', 'before', pause=True)
        submitted = await driver.command('create 2 New 73')
        try:
            self.assertEqual(submitted['result'], 0, submitted)
            await asyncio.wait_for(fault.reached.wait(), 5)
            original = submitted['pending_intents']
            identity = decode(bytes.fromhex(original[0]['bytes'])).commands[0].section
            # Force the cross-client random collision without changing native
            # bytes. The colliding peer ID is already retired at execution, and
            # the service accepts a Create of a retired identity.
            await self.update(Delta(commands=(C(1, identity, 2, 'Transient', 6), C(2, identity, 2))))
        finally:
            fault.release.set()
        committed = await driver.command('wait 3')
        self.compare(committed)
        self.assertEqual(committed['subject'], opened['subject'])
        self.assertEqual(next(node for node in committed['visible'] if node['id'] == identity)['fields']['marker']['value'], 73)
        self.assertEqual(len(self.commits()), 3)
        self.assertFalse(self.state.rejections)
        self.assertEqual(self.state.attempts['UpdateLayer'], 2)
        fresh, ready = await self.start()
        self.compare(ready)
        self.assertTrue(any(node['id'] == identity for node in ready['visible']))
        await self.close(driver)
        await self.close(fresh)

    async def live_id_collision(self, peer_first):
        # Force the cross-client random collision without changing native
        # bytes. The colliding peer ID is live at execution, and the service
        # rejects a Create of a live identity before commit. The peer's Create
        # and the rejection reach the client on different streams; the gates
        # fix which arrives first, and either order fails the layer with the
        # service's INVALID_ARGUMENT.
        driver, opened = await self.start(host=True)
        fault = self.state.plan('UpdateLayer', 'before', pause=True)
        read = None if peer_first else self.state.plan_read_frame(self.state.reads[-1]['attempt'], opened['covered'] + 1)
        for gate in (fault, read):
            if gate:
                self.addCleanup(gate.release.set)
        submitted = await driver.command('create 2 New 73')
        self.assertEqual(submitted['result'], 0, submitted)
        await asyncio.wait_for(fault.reached.wait(), 5)
        original = submitted['pending_intents']
        identity = decode(bytes.fromhex(original[0]['bytes'])).commands[0].section
        peer = await self.update(Delta(commands=(C(1, identity, 2, 'Collision', 6),)))
        self.assertEqual(peer, opened['covered'] + 1)
        if peer_first:
            # The committed peer Create applies; the pending Create of the same
            # identity is a predicted service rejection, so none of it is visible.
            applied = await driver.command(f'journal-applied {peer}')
            self.assertEqual((applied['result'], applied['status_result']), (0, 0), applied)
            self.assertEqual(applied['pending_intents'], original)
            self.assertEqual(ranked(applied['authoritative']), ranked(self.expected()))
            self.assertEqual(ranked(applied['visible']), ranked(self.expected()))
            # Host content follows the visible scene, so an export omits the predicted-rejected Create.
            self.assertEqual(applied['host'], self.export_public_fields(applied['visible']))
        else:
            await asyncio.wait_for(read.reached.wait(), 5)
        fault.release.set()
        failed = await driver.command(f'wait {peer}')
        if not peer_first:
            read.release.set()
            # The peer Create applies to the failed layer's journal and leaves the rejection's outcome.
            applied = await driver.command(f'journal-applied {peer}')
            self.assertEqual(applied['journal_applied'], peer, applied)
            self.assertEqual(ranked(applied['authoritative']), ranked(self.expected()))
            self.assertEqual({key: applied[key] for key in ('status_result', 'status_state', 'status_grpc_present')},
                             {key: failed[key] for key in ('status_result', 'status_state', 'status_grpc_present')})
        self.assertEqual(self.state.rejections, [dict(method='UpdateLayer', attempt=1, backend='A', kind='malformed',
                                                      code=grpc.StatusCode.INVALID_ARGUMENT.value[0])])
        self.assertEqual((failed['result'], failed['status_result'], failed['status_state']), (1, 1, 7))
        self.assertEqual((failed['status_grpc_present'], failed['status_grpc']),
                         (1, grpc.StatusCode.INVALID_ARGUMENT.value[0]))
        self.assertEqual(failed['subject'], opened['subject'])
        self.assertEqual(failed['pending_intents'], original)
        self.assertEqual(len(self.commits()), peer)
        self.assertEqual(self.state.attempts['UpdateLayer'], 2)  # The peer's and the rejected Create; no reissue.
        await self.close(driver)

    async def stale_reorder(self, peer_move, reorder_rule):
        # A pending Reorder of P's children names B, which a committed peer
        # update deletes or moves out of P before the service evaluates the
        # Reorder. The peer update reaches the client first; the replay skips B
        # and orders the rest, as a per-list service commits the Reorder. A
        # verbatim service rejects it instead, and that rejection fails the layer
        # with INVALID_ARGUMENT, keeps the edit and leaves the client's
        # authoritative scene the service's.
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        self.state.reorder_rule = reorder_rule
        await self.update(Delta(commands=(C(1, 100, 2, 'A', 6), C(1, 101, 2, 'B', 6), C(1, 102, 2, 'C', 6)),
                                fields=(F(100, 'marker', 17),)))
        driver, opened = await self.start(host=True)
        fault = self.state.plan('UpdateLayer', 'before', pause=True)
        self.addCleanup(fault.release.set)
        def children(state):
            return [node['name'] for node in sorted((node for node in state if node['parent'] == 2),
                                                    key=lambda node: node['order'])]
        submitted = await driver.command('reverse-children 2')
        self.assertEqual(submitted['result'], 0, submitted)
        await asyncio.wait_for(fault.reached.wait(), 5)
        original = submitted['pending_intents']
        self.assertEqual(len(original), 1)
        self.assertTrue(any(c.tag == 4 and 101 in c.children for c in decode(bytes.fromhex(original[0]['bytes'])).commands))
        self.assertEqual(children(submitted['visible']), ['C', 'B', 'A', 'Child'])
        peer = await self.update(Delta(commands=(C(3, 101, 1, 'Escaped', old_parent=2),) if peer_move else (C(2, 101, 2),)))
        applied = await driver.command(f'journal-applied {peer}')
        self.assertEqual((applied['result'], applied['status_result']), (0, 0), applied)
        self.assertEqual(applied['pending_intents'], original)
        self.assertEqual(ranked(applied['authoritative']), ranked(self.expected()))
        self.assertEqual(children(applied['authoritative']), ['Child', 'A', 'C'])
        # The replay skips B and orders the children the Reorder still names.
        self.assertEqual(children(applied['visible']), ['C', 'A', 'Child'])
        self.assertEqual(applied['host'], self.export_public_fields(applied['visible']))
        fault.release.set()
        if reorder_rule == 'per_list':
            settled = await driver.command(f'wait {peer + 1}')
            self.compare(settled)
            self.assertEqual(children(settled['visible']), ['C', 'A', 'Child'])
            self.assertFalse(self.state.rejections)
            self.assertEqual(len(self.commits()), peer + 1)
        else:
            failed = await driver.command(f'wait {peer}')
            self.assertEqual(self.state.rejections, [dict(method='UpdateLayer', attempt=2, backend='A', kind='malformed',
                                                          code=grpc.StatusCode.INVALID_ARGUMENT.value[0])])
            # A definitive rejection: FAILED with INVALID_ARGUMENT, not a malformed stream or a divergence.
            self.assertEqual((failed['result'], failed['status_result'], failed['status_state']), (1, 1, 7))
            self.assertEqual((failed['status_grpc_present'], failed['status_grpc']),
                             (1, grpc.StatusCode.INVALID_ARGUMENT.value[0]))
            self.assertEqual(failed['subject'], opened['subject'])
            self.assertEqual(failed['pending_intents'], original)
            self.assertEqual(ranked(failed['authoritative']), ranked(self.expected()))
            self.assertEqual(len(self.commits()), peer)
        # Seed, the Reorder and the peer update: the Reorder is never reissued.
        self.assertEqual(self.state.attempts['UpdateLayer'], 3)
        await self.close(driver)

    async def test_stale_reorder_after_peer_delete_fails_definitively_on_a_verbatim_service(self):
        await self.stale_reorder(False, 'verbatim')

    async def test_stale_reorder_after_peer_move_fails_definitively_on_a_verbatim_service(self):
        await self.stale_reorder(True, 'verbatim')

    async def test_stale_reorder_after_peer_delete_commits_the_prediction_on_a_per_list_service(self):
        await self.stale_reorder(False, 'per_list')

    async def test_stale_reorder_after_peer_move_commits_the_prediction_on_a_per_list_service(self):
        await self.stale_reorder(True, 'per_list')

    async def test_peer_reorder_with_repeats_and_non_children_applies_as_usd_orders_each_child_list(self):
        # A per-list service commits a peer Reorder that repeats members and names IDs that are not children;
        # every reader applies it exactly as USD's list ordering does to each child list, as the service does.
        self.state.reorder_rule = 'per_list'
        seeded = await self.update(Delta(commands=(C(1, 100, 2, 'A', 6), C(1, 101, 2, 'x', 1), C(1, 102, 2, 'B', 6),
                                                   C(1, 103, 2, 'y', 8))))
        driver, _ = await self.start(host=True)
        observer, _ = await self.start()
        peer = await self.update(Delta(commands=(C(4, 2, children=(102, 100, 102, 999, 5, 103, 101, 100, 103)),)))
        self.assertEqual(peer, seeded + 1)
        self.assertFalse(self.state.rejections)
        def children(state):
            return [node['name'] for node in sorted((node for node in state if node['parent'] == 2),
                                                    key=lambda node: node['order'])]
        # P's children are [Child, A, x, B, y]. Its prims [Child, A, B] ordered by the first occurrences [B, A]
        # become [Child, B, A] in the prims' positions, and its properties [x, y] ordered [y, x] become [y, x].
        for client in (driver, observer):
            state = await client.command(f'wait {peer}')
            self.compare(state)
            self.assertEqual(children(state['visible']), ['Child', 'B', 'y', 'A', 'x'])
        await self.close(driver)
        await self.close(observer)

    async def test_captured_create_of_a_live_id_fails_definitively_after_the_peer_create(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        await self.live_id_collision(peer_first=True)

    async def test_captured_create_of_a_live_id_fails_definitively_before_the_peer_create(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        await self.live_id_collision(peer_first=False)

    async def test_three_writers_first_creating_one_path_converge_on_the_last_commit(self):
        # Known limitation (design section 10): section identity is client-allocated, so
        # three writers that each first-create /P/World before seeing the others
        # create three sections at one (parent, name) slot. The Live Edit service
        # lets each later Create delete the occupant's subtree, so every scene
        # converges on the last committed World and the earlier Worlds and their
        # contents are deleted. No writer fails, and nothing is reissued.
        writers = [(await self.start(host=True))[0] for _ in range(3)]
        base = self.state.attempts.get('UpdateLayer', 0)
        faults = [self.state.plan('UpdateLayer', 'before', attempt=base + n, pause=True) for n in (1, 2, 3)]
        created = []
        try:
            for marker, (writer, fault) in enumerate(zip(writers, faults), 1):
                submitted = await writer.command(f'create 2 World {marker}')
                self.assertEqual(submitted['result'], 0, submitted)
                await asyncio.wait_for(fault.reached.wait(), 5)  # Issued, and held before commit.
                created.append(decode(bytes.fromhex(submitted['pending_intents'][0]['bytes'])).commands[0].section)
            first = len(self.commits())
            for offset, (writer, fault) in enumerate(zip(writers, faults), 1):
                fault.release.set()
                self.assertEqual((await writer.command(f'wait {first + offset}'))['result'], 0)  # Its own commit is applied.
        finally:
            for fault in faults:
                fault.release.set()
        final = first + 3
        for writer in writers:
            state = await writer.command(f'wait {final}')
            self.compare(state)
            worlds = [node for node in state['visible'] if node['path'] == '/P/World']
            self.assertEqual([node['id'] for node in worlds], [created[2]])
            ids = {node['id'] for node in state['visible']}
            self.assertNotIn(created[0], ids)
            self.assertNotIn(created[1], ids)
            self.assertEqual(worlds[0]['fields']['marker']['value'], 3)
        self.assertEqual(len(set(created)), 3)
        self.assertFalse(self.state.rejections)
        self.assertEqual(self.state.attempts['UpdateLayer'], base + 3)
        self.assertEqual(len(self.commits()), final)
        for writer in writers:
            await self.close(writer)

    def public_creation(self, state, prefix, amount):
        nodes = {node['path']: node for node in state}
        prim, child, attr, rel = (nodes[prefix + suffix] for suffix in ('', '/Child', '/Child.amount', '.link'))
        self.assertEqual(prim['type'], 6)
        self.assertEqual(prim['fields'], {'specifier': typed_value(Specifier(0)), 'typeName': typed_value(Token('Scope')),
                         'primChildren': typed_value(('Child',)), 'properties': typed_value(('link',))})
        # Exact authored storage from the pinned USD 25.11 constructors: an
        # untyped Def receives AnyType, while default attribute flags stay absent.
        self.assertEqual(child['fields'], {'specifier': typed_value(Specifier(0)), 'typeName': typed_value(Token('__AnyType__')),
                                          'properties': typed_value(('amount',))})
        self.assertEqual(attr['type'], 1)
        self.assertEqual(attr['fields'], {'typeName': typed_value(Token('int')), 'default': typed_value(amount)})
        self.assertEqual(rel['type'], 8)
        self.assertEqual(rel['fields'], {'custom': typed_value(True), 'variability': typed_value(Variability(1))})
        self.assertEqual((child['parent'], attr['parent'], rel['parent']), (prim['id'], child['id'], prim['id']))
        return {suffix: nodes[prefix + suffix]['id'] for suffix in ('', '/Child', '/Child.amount', '.link')}

    async def test_public_usd_creation_preserves_metadata_identity_and_reconnect_diff(self):
        other_address, other_stub = await self.backend('B')
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        address = await proxy.start()
        driver, initial = await self.start(address, host=True)
        observer, _ = await self.start()
        fault = self.state.plan('UpdateLayer', 'before', pause=True)
        submitted = await driver.command('usd-create 2 Authored 42')
        try:
            self.assertEqual(submitted['result'], 0, submitted)
            await asyncio.wait_for(fault.reached.wait(), 5)
            identities = self.public_creation(submitted['visible'], '/P/Authored', 42)
            original = submitted['pending_intents']
            self.assertEqual(len(original), 1)
            own = decode(bytes.fromhex(original[0]['bytes']))
            self.assertEqual({command.section for command in own.commands if command.tag == 1}, set(identities.values()))
            self.assertFalse(any(field.key in ('primChildren', 'properties') for field in own.fields))
            peer = await self.update(Delta(commands=(C(3, 2, 1, 'Moved', old_parent=1), C(1, 100, 2, 'Peer', 6)),
                                           fields=(F(2, 'probe', True),)))
        finally:
            fault.release.set()
        for client in (driver, observer):
            confirmed = await client.command(f'wait {peer + 1}')
            self.compare(confirmed)
            self.assertEqual(self.public_creation(confirmed['visible'], '/Moved/Authored', 42), identities)
        self.assertEqual(self.commits()[-1].payload.hex(), original[0]['bytes'])
        # Exercise typed scalars in a compacted resume response, including
        # unchanged-value/different-type fields that Python would compare equal.
        await proxy.rotate_and_wait(other_address, pause=True)
        await self.update(Delta(fields=(F(identities[''], 'specifier', Specifier(1)),
                                       F(identities[''], 'typeName', Token('Xform')),
                                       F(identities['/Child.amount'], 'default', 1), F(2, 'probe', 1))), other_stub)
        final = await self.update(Delta(fields=(F(identities['/Child.amount'], 'custom', True),
                         F(identities['/Child.amount'], 'variability', Variability(1)),
                         F(identities['.link'], 'custom', False))), other_stub)
        proxy.gate.set()
        caught = await driver.command(f'wait {final}')
        self.compare(caught)
        self.assertGreater(caught['generation'], initial['generation'])
        self.assertTrue(any(read['backend'] == 'B' and read['cursor'] == peer + 1 for read in self.state.reads))
        self.assertEqual(next(node for node in caught['visible'] if node['id'] == 2)['fields']['probe'], typed_value(1))
        self.compare(await observer.command(f'wait {final}'))
        fresh, opened = await self.start(other_address)
        self.compare(opened)
        self.assertEqual(self.state.attempts['CreateLayer'], 1)
        self.assertEqual(self.state.attempts['UpdateLayer'], 4)
        for client in (driver, observer, fresh):
            await self.close(client)

    async def test_public_usd_creation_lost_reply_retains_all_metadata_and_exact_wire(self):
        self.state.advertise_capabilities = False  # The unknown outcome is kept, never re-sent.
        driver, initial = await self.start(host=True)
        fault = self.state.plan('UpdateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        submitted = await driver.command('usd-create 2 Uncertain 73')
        self.assertEqual(submitted['result'], 0, submitted)
        await asyncio.wait_for(fault.reached.wait(), 5)
        failed = await driver.command('wait 2')
        self.assertEqual((failed['result'], failed['status_result'], failed['status_state']), (19, 19, 7))
        self.assertEqual(failed['subject'], initial['subject'])
        self.assertGreater(failed['incident'], 0)
        self.assertEqual(failed['authoritative'], initial['authoritative'])
        self.assertEqual(failed['pending_intents'], [dict(id=submitted['accepted_id'], bytes=self.commits()[-1].payload.hex())])
        identities = self.public_creation(failed['visible'], '/P/Uncertain', 73)
        fresh, opened = await self.start()
        self.compare(opened)
        self.assertEqual(self.public_creation(opened['visible'], '/P/Uncertain', 73), identities)
        self.assertEqual(self.state.attempts['UpdateLayer'], 1)
        await self.close(driver)
        await self.close(fresh)

    def public_variants(self, state, prefix, amount):
        nodes = {node['path'][len(prefix):]: node for node in state if node['path'].startswith(prefix)}
        expected = {'': (6, None), '{look=}': (11, ''), '{look=1}': (10, '{look=}'),
            '{look=high-low}': (10, '{look=}'), '{look=high-low}Child': (6, '{look=high-low}'),
            '{look=high-low}.amount': (1, '{look=high-low}'),
            '{look=high-low}{detail=}': (11, '{look=high-low}'),
            '{look=high-low}{detail=2}': (10, '{look=high-low}{detail=}'),
            '{look=high-low}{detail=2}.link': (8, '{look=high-low}{detail=2}')}
        self.assertEqual(set(nodes), set(expected))
        for suffix, (kind, parent) in expected.items():
            self.assertEqual(nodes[suffix]['type'], kind, suffix)
            if parent is not None:
                self.assertEqual(nodes[suffix]['parent'], nodes[parent]['id'], suffix)
        # Complete authored fields from the public stock-SdfData fixture;
        # Variant's prim view does not author a specifier or a second Prim ID.
        fields = {'': {'specifier': Specifier(0), 'typeName': Token('Scope'), 'variantSetChildren': ('look',)},
            '{look=}': {'variantChildren': ('1', 'high-low')}, '{look=1}': {},
            '{look=high-low}': {'primChildren': ('Child',), 'properties': ('amount',), 'variantSetChildren': ('detail',)},
            '{look=high-low}Child': {'specifier': Specifier(0), 'typeName': Token('__AnyType__')},
            '{look=high-low}.amount': {'typeName': Token('int'), 'default': amount},
            '{look=high-low}{detail=}': {'variantChildren': ('2',)},
            '{look=high-low}{detail=2}': {'properties': ('link',)},
            '{look=high-low}{detail=2}.link': {'custom': True, 'variability': Variability(1)}}
        for suffix, values in fields.items():
            self.assertEqual(nodes[suffix]['fields'], {key: typed_value(value) for key, value in values.items()}, suffix)
        return {suffix: node['id'] for suffix, node in nodes.items()}

    async def test_public_nested_variants_identity_parent_move_and_resume(self):
        other_address, other_stub = await self.backend('B')
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        driver, initial = await self.start(await proxy.start(), host=True)
        observer, _ = await self.start()
        fault = self.state.plan('UpdateLayer', 'before', pause=True)
        submitted = await driver.command('usd-variants 2 Authored 42')
        try:
            self.assertEqual(submitted['result'], 0, submitted)
            await asyncio.wait_for(fault.reached.wait(), 5)
            identities = self.public_variants(submitted['visible'], '/P/Authored', 42)
            original = submitted['pending_intents']
            self.assertEqual(len(original), 1)
            own = decode(bytes.fromhex(original[0]['bytes']))
            self.assertEqual({c.section for c in own.commands if c.tag == 1}, set(identities.values()))
            self.assertFalse(any(f.key in ('primChildren', 'properties', 'variantSetChildren', 'variantChildren')
                                 for f in own.fields))
            peer = await self.update(Delta(commands=(C(3, 2, 1, 'Moved', old_parent=1), C(1, 100, 2, 'Peer', 6))))
        finally:
            fault.release.set()
        for client in (driver, observer):
            confirmed = await client.command(f'wait {peer + 1}')
            self.compare(confirmed)
            self.assertEqual(self.public_variants(confirmed['visible'], '/Moved/Authored', 42), identities)
        self.assertEqual(self.commits()[-1].payload.hex(), original[0]['bytes'])
        await proxy.rotate_and_wait(other_address, pause=True)
        nested = identities['{look=high-low}{detail=2}']
        # Reparent a child out of a variant before deleting its original graph
        # ancestors. Resume must apply the final-state diff as a whole.
        await self.update(Delta(commands=(C(1, 101, nested, 'NewChild', 6),)), other_stub)
        survivor = identities['{look=high-low}Child']
        final = await self.update(Delta(commands=(C(3, survivor, 2, 'Survivor', old_parent=identities['{look=high-low}']),
            C(2, identities['{look=high-low}'], identities['{look=}']))), other_stub)
        proxy.gate.set()
        caught = await driver.command(f'wait {final}')
        self.compare(caught)
        self.assertGreater(caught['generation'], initial['generation'])
        self.assertTrue(any(read['backend'] == 'B' and read['cursor'] == peer + 1 for read in self.state.reads))
        self.assertEqual(next(node['path'] for node in caught['visible'] if node['id'] == survivor), '/Moved/Survivor')
        self.assertNotIn(101, {node['id'] for node in caught['visible']})
        self.compare(await observer.command(f'wait {final}'))
        fresh, opened = await self.start(other_address)
        self.compare(opened)
        self.assertEqual(self.state.attempts['CreateLayer'], 1)
        self.assertEqual(self.state.attempts['UpdateLayer'], 4)
        for client in (driver, observer, fresh):
            await self.close(client)

    async def test_public_nested_variants_lost_reply_retains_exact_intent(self):
        self.state.advertise_capabilities = False  # The unknown outcome is kept, never re-sent.
        driver, initial = await self.start(host=True)
        fault = self.state.plan('UpdateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        submitted = await driver.command('usd-variants 2 Uncertain 73')
        self.assertEqual(submitted['result'], 0, submitted)
        await asyncio.wait_for(fault.reached.wait(), 5)
        failed = await driver.command('wait 2')
        self.assertEqual((failed['result'], failed['status_result'], failed['status_state']), (19, 19, 7))
        self.assertEqual(failed['subject'], initial['subject'])
        self.assertGreater(failed['incident'], 0)
        self.assertEqual(failed['authoritative'], initial['authoritative'])
        self.assertEqual(failed['visible'], submitted['visible'])
        self.assertEqual(failed['pending_intents'], [dict(id=submitted['accepted_id'], bytes=self.commits()[-1].payload.hex())])
        identities = self.public_variants(failed['visible'], '/P/Uncertain', 73)
        fresh, opened = await self.start()
        self.compare(opened)
        self.assertEqual(self.public_variants(opened['visible'], '/P/Uncertain', 73), identities)
        self.assertEqual(self.state.attempts['UpdateLayer'], 1)
        await self.close(driver)
        await self.close(fresh)

    def public_targets(self, state, prefix, original_prefix, amount):
        paths = (original_prefix + '/Input.x', '/External.y')
        targets = (original_prefix + '/Target', '/Elsewhere')
        nodes = {node['path'][len(prefix):]: node for node in state if node['path'].startswith(prefix)}
        expected = {'': (6, None, {'specifier': Specifier(0), 'typeName': Token('Scope'), 'properties': ('amount', 'link')}),
            '.amount': (1, '', {'typeName': Token('int'), 'default': amount,
                'connectionChildren': Paths((paths[1], paths[0])), 'connectionPaths': PathOp(True, paths)}),
            '.link': (8, '', {'custom': True, 'variability': Variability(1),
                'targetChildren': Paths((targets[1], targets[0])), 'targetPaths': PathOp(True, targets)})}
        # The pinned stock proxies sort structural children, while the explicit
        # list operation preserves the caller's authored order independently.
        for path in paths:
            expected['.amount[' + path + ']'] = (2, '.amount', {})
        for path in targets:
            expected['.link[' + path + ']'] = (9, '.link', {})
        self.assertEqual(set(nodes), set(expected))
        for suffix, (kind, parent, fields) in expected.items():
            self.assertEqual(nodes[suffix]['type'], kind, suffix)
            if parent is not None:
                self.assertEqual(nodes[suffix]['parent'], nodes[parent]['id'], suffix)
            self.assertEqual(nodes[suffix]['fields'], {k: typed_value(v) for k, v in fields.items()}, suffix)
        return {suffix: node['id'] for suffix, node in nodes.items()}

    async def test_public_connections_targets_parent_move_and_resume(self):
        address, stub = await self.backend('B')
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        driver, initial = await self.start(await proxy.start(), host=True)
        fault = self.state.plan('UpdateLayer', 'before', pause=True)
        submitted = await driver.command('usd-targets 2 Authored 42')
        try:
            self.assertEqual(submitted['result'], 0, submitted)
            await asyncio.wait_for(fault.reached.wait(), 5)
            identities = self.public_targets(submitted['visible'], '/P/Authored', '/P/Authored', 42)
            original = submitted['pending_intents']
            self.assertEqual(len(original), 1)
            own = decode(bytes.fromhex(original[0]['bytes']))
            self.assertEqual({c.section for c in own.commands if c.tag == 1}, set(identities.values()))
            self.assertFalse(any(f.key in ('primChildren', 'properties', 'connectionChildren', 'targetChildren') for f in own.fields))
            peer = await self.update(Delta(commands=(C(3, 2, 1, 'Moved', old_parent=1),)))
        finally:
            fault.release.set()
        confirmed = await driver.command(f'wait {peer + 1}')
        self.compare(confirmed)
        self.assertEqual(self.public_targets(confirmed['visible'], '/Moved/Authored', '/P/Authored', 42), identities)
        self.assertEqual(self.commits()[-1].payload.hex(), original[0]['bytes'])
        await proxy.rotate_and_wait(address, pause=True)
        # All authored list buckets survive resume; their contents are separate
        # from structural child membership and may name nonexistent objects.
        op = PathOp(added=('/A',), prepended=('/P',), appended=('/Z',), deleted=('/D',), ordered=('/O',))
        await self.update(Delta(fields=(F(identities['.amount'], 'connectionPaths', op),)), stub)
        final = await self.update(Delta(fields=(F(identities['.link'], 'targetPaths', PathOp(True)),)), stub)
        proxy.gate.set()
        caught = await driver.command(f'wait {final}')
        self.compare(caught)
        self.assertGreater(caught['generation'], initial['generation'])
        self.assertTrue(any(r['backend'] == 'B' and r['cursor'] == peer + 1 for r in self.state.reads))
        actual = {n['id']: n for n in caught['visible']}
        self.assertEqual(actual[identities['.amount']]['fields']['connectionPaths'], typed_value(op))
        self.assertEqual(actual[identities['.link']]['fields']['targetPaths'], typed_value(PathOp(True)))
        fresh, opened = await self.start(address)
        self.compare(opened)
        self.assertEqual(self.state.attempts['UpdateLayer'], 4)
        await self.close(driver)
        await self.close(fresh)

    async def test_public_connections_targets_lost_reply_preserves_graph(self):
        self.state.advertise_capabilities = False  # The unknown outcome is kept, never re-sent.
        driver, initial = await self.start(host=True)
        fault = self.state.plan('UpdateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        submitted = await driver.command('usd-targets 2 Uncertain 73')
        self.assertEqual(submitted['result'], 0, submitted)
        await asyncio.wait_for(fault.reached.wait(), 5)
        failed = await driver.command('wait 2')
        self.assertEqual((failed['result'], failed['status_result'], failed['status_state']), (19, 19, 7))
        self.assertEqual(failed['subject'], initial['subject'])
        self.assertGreater(failed['incident'], 0)
        self.assertEqual(failed['authoritative'], initial['authoritative'])
        self.assertEqual(failed['visible'], submitted['visible'])
        self.assertEqual(failed['pending_intents'], [dict(id=submitted['accepted_id'], bytes=self.commits()[-1].payload.hex())])
        identities = self.public_targets(failed['visible'], '/P/Uncertain', '/P/Uncertain', 73)
        fresh, opened = await self.start()
        self.compare(opened)
        self.assertEqual(self.public_targets(opened['visible'], '/P/Uncertain', '/P/Uncertain', 73), identities)
        self.assertEqual(self.state.attempts['UpdateLayer'], 1)
        await self.close(driver)
        await self.close(fresh)

    async def test_path_child_policies_field_replay_and_reconnect_survivor(self):
        address, stub = await self.backend('B')
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        driver, initial = await self.start(await proxy.start(), host=True)
        seeded = await self.update(Delta(commands=(C(1, 100, 2, 'a', 1), C(1, 101, 100, '/T.x', 2),
            C(1, 102, 100, 'expression', 3), C(1, 103, 100, '/M.x', 4), C(1, 104, 103, 'arg', 5),
            C(1, 105, 2, 'rel', 8), C(1, 106, 105, '/T', 9), C(1, 107, 106, 'weight', 1),
            C(1, 108, 105, 'expression', 3)), fields=(F(107, 'default', 42),)))
        self.compare(await driver.command(f'wait {seeded}'))
        fault = self.state.plan('UpdateLayer', 'before', pause=True)
        submitted = await driver.command('set 107 marker 73')
        try:
            self.assertEqual(submitted['result'], 0, submitted)
            await asyncio.wait_for(fault.reached.wait(), 5)
            original = submitted['pending_intents']
            self.assertEqual(len(original), 1)
            peer = await self.update(Delta(commands=(C(1, 109, 106, 'peer', 1), C(3, 2, 1, 'Moved', old_parent=1))))
        finally:
            fault.release.set()
        confirmed = await driver.command(f'wait {peer + 1}')
        self.compare(confirmed)
        self.assertEqual(self.commits()[-1].payload.hex(), original[0]['bytes'])
        nodes = {n['id']: n for n in confirmed['visible']}
        self.assertEqual(nodes[107]['path'], '/Moved.rel[/T].weight')
        self.assertEqual(nodes[107]['fields'], {'default': typed_value(42), 'marker': typed_value(73)})
        self.assertEqual(nodes[104]['path'], '/Moved.a.mapper[/M.x].arg')
        self.assertEqual(nodes[103]['name'], '/M.x')
        await proxy.rotate_and_wait(address, pause=True)
        await self.update(Delta(commands=(C(1, 110, 106, 'laterPeer', 1),)), stub)
        final = await self.update(Delta(commands=(C(3, 107, 2, 'survivor', old_parent=106), C(2, 105, 2))), stub)
        proxy.gate.set()
        caught = await driver.command(f'wait {final}')
        self.compare(caught)
        self.assertGreater(caught['generation'], initial['generation'])
        self.assertTrue(any(r['backend'] == 'B' and r['cursor'] == peer + 1 for r in self.state.reads))
        nodes = {n['id']: n for n in caught['visible']}
        self.assertEqual(nodes[107]['path'], '/Moved.survivor')
        self.assertEqual(nodes[107]['fields'], {'default': typed_value(42), 'marker': typed_value(73)})
        self.assertFalse(set(nodes) & {105, 106, 108, 109, 110})
        self.assertEqual(nodes[104]['path'], '/Moved.a.mapper[/M.x].arg')
        fresh, opened = await self.start(address)
        self.compare(opened)
        self.assertEqual(self.state.attempts['UpdateLayer'], 5)
        await self.close(driver)
        await self.close(fresh)

    async def test_seeded_structural_writers_account_deleted_and_overwritten_edits(self):
        import structural_history as history
        plan = history.generate(self.seed, clients=getattr(self, 'writer_clients', 4))
        from tools.live.writer_campaign import observe_workload
        observe_workload(self, plan)
        stem = self.id().split('.')[-1]
        evidence = dict(plan=plan, admissions=[], peers=[], rounds=[], commits=[])
        self.trace['structural_history'] = evidence
        evidence_path = self.artifact / (stem + '.history.json')
        print(json.dumps({'structural_master_seed': plan['master_seed'], 'structural_writer_seeds': plan['writer_seeds']}), flush=True)

        async def together(jobs):
            tasks = [asyncio.create_task(job) for job in jobs]
            try:
                return await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        def persist():
            evidence_path.write_text(json.dumps(evidence, indent=2) + '\n')

        def ledger():
            return [dict(sequence=c.sequence, writer=c.writer, bytes=c.payload.hex()) for c in self.commits()[1:]]

        def expected(rounds):
            evidence['commits'] = ledger()
            persist()  # Keep the observed failing prefix even if accounting rejects it.
            result = history.account(plan, evidence['admissions'], evidence['peers'], evidence['commits'], rounds)
            evidence['accounting'] = result
            persist()
            return result

        async def peer(effects, stub=None):
            commands, fields = [], []
            for effect in effects:
                if effect['kind'] == 'move':
                    commands.append(C(3, effect['id'], effect['parent'], effect['name'], old_parent=effect['old_parent']))
                elif effect['kind'] == 'create':
                    commands.append(C(1, effect['id'], effect['parent'], effect['name'], 6))
                    fields.append(F(effect['id'], 'marker', effect['value']))
                else:
                    fields.append(F(effect['id'], effect['key'], effect['value']))
            delta = Delta(commands=tuple(commands), fields=tuple(fields))
            evidence['peers'].append(dict(bytes=encode(delta).hex(), effects=effects))
            persist()  # Chosen effects are recorded before issuing the request.
            return await self.update(delta, stub)

        persist()  # Master/writer seeds and all inputs precede native processes.
        address_b, stub_b = await self.backend('B')
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        address = await proxy.start()
        frontier = await peer([dict(kind='create', id=1000, parent=5, name='Sentinel', value=1)])
        oracle = expected(0)
        drivers = [await self.start(address, host=True) for _ in range(plan['client_count'])]
        for _, opened in drivers:
            self.compare(opened)
            history.verify_scene(opened['visible'], oracle['scene'])
        generations = [opened['generation'] for _, opened in drivers]
        active_stub = self.stub
        for ordinal in range(plan['rounds']):
            ops = [writer[ordinal] for writer in plan['writers']]
            nodes = {node['id']: node for node in oracle['scene']}
            commands, prefixes = [], {}
            for op in ops:
                branch = oracle['branches'].get(op['writer'])
                if op['kind'] == 'create':
                    prefixes[op['writer']] = nodes[op['parent']]['path'] + '/' + op['name']
                    commands.append(f"{op['shape']} {op['parent']} {op['name']} {op['value']}")
                elif op['kind'] == 'field':
                    commands.append(f"set {history.leaf(branch)} default {op['value']}")
                elif op['kind'] == 'shared':
                    commands.append(f"set 2 shared {op['value']}")
                elif op['kind'] == 'move':
                    commands.append(f"move {branch['ids']['']} {nodes[op['parent']]['path']}/{op['name']}")
                else:
                    commands.append(f"erase {branch['ids']['']}")
            first_attempt = self.state.attempts.get('UpdateLayer', 0) + 1
            faults = [self.state.plan('UpdateLayer', 'before', attempt=first_attempt + i, pause=True)
                      for i in range(plan['client_count'])]
            record = dict(ordinal=ordinal, commands=commands, first_attempt=first_attempt,
                          release_order=plan['release_orders'][ordinal])
            evidence['rounds'].append(record)
            persist()
            try:
                results = await together(driver.command(command) for (driver, _), command in zip(drivers, commands))
                for result in results:
                    self.assertEqual(result['result'], 0, result)
                await asyncio.wait_for(asyncio.gather(*(f.reached.wait() for f in faults)), 5)
                for op, result in zip(ops, results):
                    self.assertEqual(len(result['pending_intents']), 1, result)
                    intent = result['pending_intents'][0]
                    self.assertEqual(intent['id'], result['accepted_id'])
                    admission = dict(edit=op['edit'], accepted_id=intent['id'], bytes=intent['bytes'])
                    if op['kind'] == 'create':
                        prefix = prefixes[op['writer']]
                        specs = history.template(op['shape'], prefix, op['value'])
                        authored = {node['path']: node for node in result['visible']}
                        admission.update(captured_prefix=prefix, ids={suffix: authored[prefix + suffix]['id'] for suffix in specs})
                        for suffix, (kind, parent_suffix, fields) in specs.items():
                            node = authored[prefix + suffix]
                            self.assertEqual((node['type'], node['fields']), (kind, fields))
                            self.assertEqual(node['parent'], op['parent'] if parent_suffix is None else admission['ids'][parent_suffix])
                    evidence['admissions'].append(admission)
                persist()
                effects = [dict(kind='field', id=1, key='epoch', value=ordinal)]
                if ordinal in (0, 2, 7):
                    effects.append(dict(kind='move', id=2, parent=1, old_parent=1, name=f'Parent{ordinal}'))
                if ops[0]['kind'] != 'create':
                    writer = ordinal % plan['client_count']
                    effects.append(dict(kind='create', id=2000 + ordinal,
                        parent=oracle['branches'][writer]['ids'][''], name=f'Late{ordinal}', value=ordinal + 500))
                peer_frontier = await peer(effects, active_stub)
            finally:
                for slot in record['release_order']:
                    faults[slot].release.set()
            frontier = peer_frontier + plan['client_count']
            reports = await together(driver.command(f'wait {frontier}') for driver, _ in drivers)
            for report in reports:
                self.assertEqual(report['result'], 0, report)
            oracle = expected(ordinal + 1)
            for report in reports:
                self.compare(report)
                history.verify_scene(report['visible'], oracle['scene'])
            record['confirmed_frontier'] = frontier
            if ordinal in plan['rotation_after']:
                generations = [report['generation'] for report in reports]
                destination, active_stub, backend = (address_b, stub_b, 'B') if ordinal == 5 else (self.address, self.stub, 'A')
                before = frontier
                await proxy.rotate_and_wait(destination, pause=True)
                frontier = await peer([dict(kind='field', id=1, key='epoch', value=100 + ordinal)], active_stub)
                proxy.gate.set()
                reports = await together(driver.command(f'wait {frontier}') for driver, _ in drivers)
                oracle = expected(ordinal + 1)
                for i, report in enumerate(reports):
                    self.compare(report)
                    history.verify_scene(report['visible'], oracle['scene'])
                    self.assertGreater(report['generation'], generations[i])
                    generations[i] = report['generation']
                self.assertGreaterEqual(sum(r['backend'] == backend and r['cursor'] == before for r in self.state.reads), plan['client_count'])
                record['rotation'] = dict(backend=backend, cursor=before, resumed=frontier)
            persist()
        fresh, opened = await self.start()
        self.compare(opened)
        history.verify_scene(opened['visible'], oracle['scene'])
        self.assertEqual(len(oracle['lineage']), plan['client_count'] * plan['rounds'])
        self.assertTrue(any(entry['deleted_ids'] for entry in oracle['lineage'].values()))
        self.assertTrue(any('superseded_by' in entry for entry in oracle['lineage'].values()))
        self.assertEqual(self.state.attempts['CreateLayer'], 1)
        self.assertEqual(self.state.attempts['UpdateLayer'], len(evidence['admissions']) + len(evidence['peers']))
        for driver, _ in drivers:
            await self.close(driver)
        await self.close(fresh)

    async def test_seeded_sample_writers_account_repeated_and_erased_edits(self):
        import sample_history as history
        from tools.live.test_support import write_json
        plan = history.generate(self.seed, clients=getattr(self, 'writer_clients', 4),
                                rounds=getattr(self, 'writer_steps', 16))
        from tools.live.writer_campaign import observe_workload
        observe_workload(self, plan)
        stem = self.id().split('.')[-1]
        evidence = dict(plan=plan, bindings=[], admissions=[], peers=[], rounds=[], commits=[])
        self.trace['sample_history'] = evidence
        evidence_path = self.artifact / (stem + '.history.json')
        print(json.dumps({'sample_master_seed': plan['master_seed'], 'sample_writer_seeds': plan['writer_seeds']}), flush=True)

        async def together(jobs):
            tasks = [asyncio.create_task(job) for job in jobs]
            try:
                return await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks: task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        def persist():
            write_json(evidence_path, evidence)

        def decoded_effect(payload):
            delta = decode(bytes.fromhex(payload))
            return dict(is_diff=delta.is_diff, base_version=delta.base_version, source_format=delta.source_format,
                        commands=len(delta.commands), fields=[dict(section=f.section, key=f.key, set_order=f.set_order,
                            value=typed_value(f.value) if f.value is not None else None) for f in delta.fields],
                        samples=[dict(section=s.section, time_bits=history.bits(s.time),
                            value=typed_value(s.value) if s.value is not None else None) for s in delta.samples])

        def expected(rounds):
            evidence['commits'] = [dict(sequence=c.sequence, writer=c.writer, bytes=c.payload.hex())
                                   for c in self.commits()[1:]]
            persist()
            result = history.account(plan, evidence['bindings'], evidence['admissions'], evidence['peers'],
                                     evidence['commits'], rounds)
            evidence['accounting'] = result
            persist()
            return result

        def verify(report, oracle, rounds):
            self.compare(report)
            history.verify_scene(report['visible'], oracle['scene'])
            self.assertEqual((report['local_admitted'], report['local_retired']), (rounds, rounds))

        persist()  # Inputs and seeds precede native process startup.
        address_b, stub_b = await self.backend('B')
        proxy = RotatingProxy(self.address); self.proxies.append(proxy)
        address = await proxy.start()
        drivers = [await self.start(address, host=True) for _ in range(plan['client_count'])]
        evidence['bindings'] = [opened['writer'] for _, opened in drivers]
        persist()  # Native writer identity is observed before the first admission.
        oracle = expected(0)
        for _, opened in drivers: verify(opened, oracle, 0)
        frontier = 1
        for ordinal in range(plan['rounds']):
            operations = [writer[ordinal] for writer in plan['writers']]
            commands = [f"{op['kind']} {op['section']} {op['time_bits']}" +
                        ('' if op['value'] is None else f" {op['value']}") for op in operations]
            first_attempt = self.state.attempts.get('UpdateLayer', 0) + 1
            faults = [self.state.plan('UpdateLayer', 'before', attempt=first_attempt+i, pause=True)
                      for i in range(plan['client_count'])]
            record = dict(ordinal=ordinal, commands=commands, first_attempt=first_attempt,
                          release_order=plan['release_orders'][ordinal])
            evidence['rounds'].append(record); persist()
            try:
                reports = await together(driver.command(command) for (driver, _),command in zip(drivers,commands))
                await asyncio.wait_for(asyncio.gather(*(fault.reached.wait() for fault in faults)), 5)
                for op, report in zip(operations, reports):
                    self.assertEqual(report['result'], 0, report)
                    self.assertEqual(len(report['pending_intents']), 1, report)
                    intent = report['pending_intents'][0]
                    self.assertEqual(intent['id'], report['accepted_id'])
                    evidence['admissions'].append(dict(edit=op['edit'], writer=report['writer'],
                        accepted_id=intent['id'], local_ordinal=report['local_admitted'], bytes=intent['bytes']))
                persist()  # Exact accepted occurrences and bytes precede release.
                for op, admission in zip(operations, evidence['admissions'][-plan['client_count']:]):
                    admission['effect'] = decoded_effect(admission['bytes'])
                    persist()  # Preserve a miscompiled losing edit before rejecting it.
                    history.verify_effect(op, admission['effect'],
                                          history.sample_keys(oracle['scene'], op['section']))
            finally:
                for slot in record['release_order']: faults[slot].release.set()
            frontier += plan['client_count']
            reports = await together(driver.command(f'wait {frontier}') for driver, _ in drivers)
            oracle = expected(ordinal+1)
            for report in reports: verify(report, oracle, ordinal+1)
            record['confirmed_frontier'] = frontier
            if ordinal in plan['rotation_after']:
                destination, stub, backend = (address_b, stub_b, 'B') if ordinal == 5 else (self.address, self.stub, 'A')
                cursor = frontier
                await proxy.rotate_and_wait(destination, pause=True)
                stamp = -0.0 if ordinal == 5 else 2.0
                peer = dict(edit=f'peer:{ordinal}', after_round=ordinal, kind='sample', section=6,
                            time_bits=history.bits(stamp), value=1000+ordinal)
                delta = Delta(samples=(S(6, stamp, peer['value']),))
                peer['bytes'] = encode(delta).hex()
                evidence['peers'].append(peer); persist()
                history.verify_effect(peer, decoded_effect(peer['bytes']))
                frontier = await self.update(delta, stub)
                proxy.gate.set()
                resumed = await together(driver.command(f'wait {frontier}') for driver, _ in drivers)
                oracle = expected(ordinal+1)
                for previous, report in zip(reports, resumed):
                    verify(report, oracle, ordinal+1)
                    self.assertGreater(report['generation'], previous['generation'])
                self.assertGreaterEqual(sum(r['backend']==backend and r['cursor']==cursor for r in self.state.reads),
                                        plan['client_count'])
                record['rotation'] = dict(backend=backend, cursor=cursor, resumed=frontier)
            persist()
        fresh, opened = await self.start()
        self.compare(opened)
        history.verify_scene(opened['visible'], oracle['scene'])
        self.assertEqual(oracle['native_commits'], plan['client_count']*plan['rounds'])
        self.assertEqual(oracle['peer_commits'], 2)
        self.assertTrue(any(item['outcome']=='no_effect' for item in oracle['lineage'].values()))
        self.assertTrue(any(item['outcome']=='erased' for item in oracle['lineage'].values()))
        self.assertTrue(any(item['outcome']=='superseded' for item in oracle['lineage'].values()))
        self.assertEqual(self.state.attempts['UpdateLayer'], len(evidence['admissions'])+len(evidence['peers']))
        for driver, _ in drivers: await self.close(driver)
        await self.close(fresh)

    async def test_captured_rename_reorder_naming_moved_sibling_is_rejected_definitively(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        seeded = await self.update(Delta(commands=(C(1, 100, 2, 'A', 6), C(1, 101, 2, 'B', 6), C(1, 102, 2, 'C', 6)),
                                        fields=(F(100, 'marker', 17), F(101, 'marker', 23))))
        driver, initial = await self.start(host=True)
        read_gate = self.state.plan_read_frame(self.state.reads[-1]['attempt'], seeded + 1)
        self.addCleanup(read_gate.release.set)
        fault = self.state.plan('UpdateLayer', 'before', pause=True)
        submitted = await driver.command('move 100 /P/Renamed')
        try:
            self.assertEqual(submitted['result'], 0, submitted)
            await asyncio.wait_for(fault.reached.wait(), 5)
            original = submitted['pending_intents']
            self.assertEqual(len(original), 1)
            wire = decode(bytes.fromhex(original[0]['bytes']))
            self.assertTrue(any(c.tag == 4 and 101 in c.children for c in wire.commands))
            peer = await self.update(Delta(commands=(C(3, 101, 5, 'Escaped', old_parent=2),)))
            # Hold peer delivery while the service evaluates the original edit.
            await asyncio.wait_for(read_gate.reached.wait(), 5)
        finally:
            fault.release.set()
        async def rejected():
            async with self.state.changed:
                await self.state.changed.wait_for(lambda: self.state.rejections)
        await asyncio.wait_for(rejected(), 5)
        read_gate.release.set()
        # The service rejects the whole update: the Reorder names a sibling that
        # the peer moved away, and the original bytes are never committed.
        self.assertEqual(self.state.rejections, [dict(method='UpdateLayer', attempt=2, backend='A', kind='malformed',
                                                      code=grpc.StatusCode.INVALID_ARGUMENT.value[0])])
        self.assertEqual(len(self.commits()), peer)
        failed = await driver.command(f'wait {peer}')
        self.assertEqual(failed['pending_intents'], original)
        self.assertEqual(failed['pending'], 1)
        self.assertEqual(failed['subject'], initial['subject'])
        # A definitive rejection latches FAILED with INVALID_ARGUMENT and keeps
        # the gRPC code (design section 13). Nothing else is admitted or reissued.
        self.assertEqual((failed['result'], failed['status_result'], failed['status_state']), (1, 1, 7))
        self.assertEqual((failed['status_grpc_present'], failed['status_grpc']),
                         (1, grpc.StatusCode.INVALID_ARGUMENT.value[0]))
        # Seed, original rename and peer move: the rejected edit is never reissued.
        self.assertEqual(self.state.attempts['UpdateLayer'], 3)
        self.assertEqual(len(self.state.rejections), 1)
        await self.close(driver)

    @staticmethod
    def export_public_fields(scene):
        # Pinned USD public field lists expose required schema fallbacks.
        result = []
        for node in sorted(scene, key=lambda n: n['path']):
            fields = dict(node['fields'])
            if node['type'] in (6, 10):
                fields.setdefault('specifier', {'type': 'specifier', 'value': 1})
            elif node['type'] == 1:
                fields.setdefault('custom', {'type': 'bool', 'value': False})
                fields.setdefault('typeName', {'type': 'token', 'value': ''})
                fields.setdefault('variability', {'type': 'variability', 'value': 0})
            # Stock SdfData and the live host backing both store connectionChildren
            # and targetChildren in SdfPath order; a crate or text reread derives
            # them in list-operation order, so the offline graph is compared as
            # path sets.
            for key in ('connectionChildren', 'targetChildren'):
                if key in fields:
                    fields[key] = dict(fields[key], value=sorted(fields[key]['value']))
            result.append(dict(path=node['path'], type=node['type'], fields=fields))
        return result

    @staticmethod
    async def together(jobs):
        tasks = [asyncio.create_task(job) for job in jobs]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def offline(self, executable, *args):
        process = await asyncio.create_subprocess_exec(str(executable), *map(str, args),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        self.assertEqual(process.returncode, 0, stderr.decode(errors='replace'))
        return json.loads(stdout)


    async def test_structural_writers_export_unknown_and_dependent_suffix_offline(self):
        self.state.advertise_capabilities = False  # The unknown outcome is kept, never re-sent.
        import structural_history as history
        public_fields = self.export_public_fields
        together = self.together
        offline = self.offline

        directory = self.artifact / (self.id().split('.')[-1] + '-export')
        directory.mkdir()
        rng = random.Random(self.seed)
        shapes = ('usd-create', 'usd-variants', 'usd-targets')
        amounts = [rng.randrange(1000, 2000) for _ in shapes]
        self.trace['structural_export_plan'] = dict(seed=self.seed, shapes=shapes, values=amounts)
        print(json.dumps({'structural_export_seed': self.seed, 'values': amounts}), flush=True)
        # The generic wire fixture intentionally omits attribute type names.
        # This USD export case needs a stock-parseable host scene.
        base = await self.update(Delta(fields=(F(4, 'typeName', Token('int')), F(6, 'typeName', Token('int')))))
        drivers = [await self.start(host=True) for _ in shapes]
        for _, initial in drivers:
            self.compare(initial)
        # Hold all first writes before execution, and all replies after commit.
        # Reads may apply the independent peer first, but none of these commits.
        first_attempt = self.state.attempts.get('UpdateLayer', 0) + 1
        before = [self.state.plan('UpdateLayer', 'before', attempt=first_attempt + i, pause=True) for i in range(3)]
        after = [self.state.plan('UpdateLayer', 'after_commit', attempt=first_attempt + i,
                                 pause=True, code=grpc.StatusCode.UNAVAILABLE) for i in range(3)]
        reads = [self.state.plan_read_frame(read['attempt'], base + 2, include_suppressed=True) for read in self.state.reads[-3:]]
        for gate in before + after + reads:
            self.addCleanup(gate.release.set)
        submitted = await together(driver.command(f'{shape} 2 Writer{i} {amounts[i]}')
                                  for i, ((driver, _), shape) in enumerate(zip(drivers, shapes)))
        await asyncio.wait_for(asyncio.gather(*(gate.reached.wait() for gate in before)), 5)
        branches = []
        for i, state in enumerate(submitted):
            self.assertEqual(state['result'], 0, state)
            specs = history.template(shapes[i], f'/P/Writer{i}', amounts[i])
            nodes = {n['path']: n for n in state['visible']}
            ids = {suffix: nodes[f'/P/Writer{i}' + suffix]['id'] for suffix in specs}
            branches.append(dict(ids=ids, shape=shapes[i]))
            for suffix, (kind, parent_suffix, fields) in specs.items():
                node = nodes[f'/P/Writer{i}' + suffix]
                self.assertEqual((node['type'], node['fields']), (kind, fields))
                self.assertEqual(node['parent'], 2 if parent_suffix is None else ids[parent_suffix])
        moved = await together(driver.command(f"move {branch['ids']['']} /Q/Queued{i}")
                               for i, ((driver, _), branch) in enumerate(zip(drivers, branches)))
        queued = await together(driver.command(f"set {history.leaf(branch)} default {amounts[i] + 10000}")
                                for i, ((driver, _), branch) in enumerate(zip(drivers, branches)))
        for i, state in enumerate(queued):
            self.assertEqual(state['result'], 0, state)
            self.assertEqual([p['id'] for p in state['pending_intents']],
                             [submitted[i]['accepted_id'], moved[i]['accepted_id'], state['accepted_id']])
        peer = await self.update(Delta(commands=(C(3, 2, 1, 'PeerParent', old_parent=1), C(1, 100, 3, 'Late', 6)),
                                       fields=(F(100, 'marker', 73),)))
        self.assertEqual(peer, base + 1)
        retained = await together(driver.command(f'journal-applied {peer}') for driver, _ in drivers)
        original = [state['pending_intents'] for state in queued]
        for state, intents in zip(retained, original):
            self.assertEqual(state['result'], 0, state)
            self.assertEqual(state['pending_intents'], intents)
            self.assertEqual(state['authoritative'], self.expected())
            self.assertEqual(state['host'], public_fields(state['visible']))
        self.trace['structural_export_retained'] = retained
        for gate in before:
            gate.release.set()
        await asyncio.wait_for(asyncio.gather(*(gate.reached.wait() for gate in after + reads)), 5)
        self.assertEqual(len(self.commits()), peer + 3)
        for gate in after:
            gate.release.set()
        failed = await together(driver.command(f'wait {peer + 3}') for driver, _ in drivers)
        for i, state in enumerate(failed):
            self.assertEqual((state['result'], state['status_result'], state['status_state']), (19, 19, 7))
            self.assertEqual(state['subject'], drivers[i][1]['subject'])
            self.assertGreater(state['incident'], 0)
            for key in ('pending_intents', 'authoritative', 'visible', 'host'):
                self.assertEqual(state[key], retained[i][key])
        commits = self.commits()[peer:]
        self.assertEqual(sorted(c.payload.hex() for c in commits), sorted(p[0]['bytes'] for p in original))
        self.assertEqual(len({c.writer for c in commits}), 3)
        self.assertEqual(self.state.attempts['UpdateLayer'], 5)  # Preparation, independent peer, three first writes.
        for gate in reads:
            gate.release.set()
        fresh, opened = await self.start(host=True)
        self.compare(opened)
        paths = {node['id']: node['path'] for node in opened['visible']}
        for i, branch in enumerate(branches):
            self.assertEqual(paths[branch['ids']['']], f'/PeerParent/Writer{i}')
            value_node = next(n for n in opened['visible'] if n['id'] == history.leaf(branch))
            self.assertEqual(value_node['fields']['default'], {'type': 'int32', 'value': amounts[i]})
        attempts = dict(self.state.attempts)
        for i, (driver, _) in enumerate(drivers):
            exported = await driver.command(f'export "{(directory / f"writer-{i}.usdc").as_posix()}"')
            self.assertEqual(exported['result'], 0, exported)
            self.assertEqual(exported['pending_intents'], original[i])
        self.assertEqual(self.state.attempts, attempts, 'export performed network work')
        for driver, _ in drivers:
            await self.close(driver)
        await self.close(fresh)
        # All original process, USD, journal and runtime owners have exited.
        executable = Path(os.environ['LIVE_TEST_SCENE_DRIVER'])
        for i in range(3):
            graph = await offline(executable, '--usd-graph', directory / f'writer-{i}.usdc')
            self.assertEqual(public_fields(graph['host']), public_fields(retained[i]['host']))
        self.assertEqual(self.state.attempts, attempts)

    async def test_sample_writers_export_unknown_and_overwritten_suffix_offline(self):
        self.state.advertise_capabilities = False  # The unknown outcome is kept, never re-sent.
        import sample_history as history
        from tools.live.test_support import write_json
        together, offline = self.together, self.offline
        directory = self.artifact / (self.id().split('.')[-1] + '-export')
        directory.mkdir()
        rng = random.Random(self.seed)
        plans = []
        for head in ('sample', 'sample-map', 'erase-sample'):
            amount = rng.randrange(1000, 2000)
            operations = [(head, 1.0 if head == 'erase-sample' else -0.0, amount),
                          ('sample', 0.0, amount + 1), ('sample-map', -0.0, amount + 2),
                          ('erase-sample', 0.0, None), ('erase-sample', 0.0, None),
                          ('sample', 2.0, amount + 3)]
            plans.append([dict(kind=kind, section=4, time_bits=history.bits(stamp),
                               value=None if kind == 'erase-sample' else value)
                          for kind, stamp, value in operations])
        write_json(directory / 'plan.json', dict(seed=self.seed, writers=plans))
        print(json.dumps({'sample_export_seed': self.seed, 'writers': plans}), flush=True)
        base = await self.update(Delta(fields=(F(4, 'typeName', Token('int')), F(6, 'typeName', Token('int')),
                                               F(4, 'timeSamples', T(((1.0, 31), (2.0, 32)))))))
        drivers = [await self.start(host=True) for _ in plans]
        for _, initial in drivers:
            self.compare(initial)
        first = self.state.attempts.get('UpdateLayer', 0) + 1
        before = [self.state.plan('UpdateLayer', 'before', attempt=first + i, pause=True) for i in range(3)]
        after = [self.state.plan('UpdateLayer', 'after_commit', attempt=first + i,
                                 pause=True, code=grpc.StatusCode.UNAVAILABLE) for i in range(3)]
        reads = [self.state.plan_read_frame(r['attempt'], base + 2, include_suppressed=True)
                 for r in self.state.reads[-3:]]
        for gate in before + after + reads:
            self.addCleanup(gate.release.set)
        original, authored = [[] for _ in plans], [[] for _ in plans]
        models = [Model() for _ in plans]
        for model in models:
            for commit in self.commits():
                model.apply(commit.payload, commit.sequence)
        for ordinal in range(len(plans[0])):
            commands = [f"{p[ordinal]['kind']} 4 {p[ordinal]['time_bits']}" +
                        ('' if p[ordinal]['value'] is None else f" {p[ordinal]['value']}") for p in plans]
            reports = await together(driver.command(command) for (driver, _), command in zip(drivers, commands))
            if ordinal == 0:
                await asyncio.wait_for(asyncio.gather(*(gate.reached.wait() for gate in before)), 5)
            for i, report in enumerate(reports):
                self.assertEqual(report['result'], 0, report)
                pending = report['pending_intents']
                self.assertEqual(pending[:-1], original[i])
                self.assertEqual(pending[-1]['id'], report['accepted_id'])
                self.assertGreater(report['accepted_id'], 0)
                self.assertEqual(len({p['id'] for p in pending}), ordinal + 1)
                original[i] = pending
                # Check each effect against authored input before any head can
                # commit, including values/maps later overwritten and no-op erases.
                delta = decode(bytes.fromhex(pending[-1]['bytes']))
                effect = dict(is_diff=delta.is_diff, base_version=delta.base_version, source_format=delta.source_format,
                    commands=len(delta.commands), fields=[dict(section=f.section, key=f.key, set_order=f.set_order,
                        value=None if f.value is None else typed_value(f.value)) for f in delta.fields],
                    samples=[dict(section=s.section, time_bits=history.bits(s.time),
                        value=None if s.value is None else typed_value(s.value)) for s in delta.samples])
                record = dict(id=report['accepted_id'], base=typed(models[i].state), effect=effect,
                              provenance=report['intent_provenance'][-1])
                authored[i].append(record)
                write_json(directory / f'writer-{i}.admissions.json', dict(original=pending, authored=authored[i]))
                history.verify_effect(plans[i][ordinal], effect, history.sample_keys(record['base'], 4))
                models[i].apply(bytes.fromhex(pending[-1]['bytes']))
                record['visible'] = typed(models[i].state)
                self.assertEqual(report['visible'], record['visible'])
                self.assertEqual((report['local_admitted'], report['local_retired']), (ordinal + 1, 0))
        peer = await self.update(Delta(samples=(S(4, 3.0, 303), S(6, -0.0, 606))))
        self.assertEqual(peer, base + 1)
        retained = await together(driver.command(f'journal-applied {peer}') for driver, _ in drivers)
        for i, report in enumerate(retained):
            model = Model()
            for commit in self.commits():
                model.apply(commit.payload, commit.sequence)
            for entry in original[i]:
                model.apply(bytes.fromhex(entry['bytes']))
            self.assertEqual(report['result'], 0, report)
            self.assertEqual(report['pending_intents'], original[i])
            self.assertEqual(report['intent_provenance'], [r['provenance'] for r in authored[i]])
            self.assertEqual(report['authoritative'], self.expected())
            self.assertEqual(report['visible'], typed(model.state))
            self.assertEqual(report['host'], self.export_public_fields(report['visible']))
        self.trace['sample_export_retained'] = retained
        for gate in before:
            gate.release.set()
        await asyncio.wait_for(asyncio.gather(*(gate.reached.wait() for gate in after + reads)), 5)
        commits = self.commits()[peer:]
        self.assertEqual(len(commits), 3)
        self.assertEqual({c.writer: c.payload.hex() for c in commits},
                         {initial['writer']: original[i][0]['bytes'] for i, (_, initial) in enumerate(drivers)})
        for gate in after:
            gate.release.set()
        failed = await together(driver.command(f'wait {peer + 3}') for driver, _ in drivers)
        for i, report in enumerate(failed):
            self.assertEqual((report['result'], report['status_result'], report['status_state']), (19, 19, 7))
            self.assertEqual(report['subject'], drivers[i][1]['subject'])
            self.assertGreater(report['incident'], 0)
            for key in ('pending_intents', 'intent_provenance', 'authoritative', 'visible', 'host', 'local_admitted', 'local_retired'):
                self.assertEqual(report[key], retained[i][key])
        for gate in reads:
            gate.release.set()
        fresh, opened = await self.start(host=True)
        self.compare(opened)
        self.assertEqual(self.state.attempts['UpdateLayer'], 5)
        attempts = dict(self.state.attempts)
        for i, (driver, _) in enumerate(drivers):
            exported = await driver.command(f'export "{(directory / f"writer-{i}.usdc").as_posix()}"')
            self.assertEqual(exported['result'], 0, exported)
            self.assertEqual(exported['pending_intents'], original[i])
            write_json(directory / f'writer-{i}.admissions.json', dict(original=original[i], authored=authored[i]))
        self.assertEqual(self.state.attempts, attempts)
        for driver, _ in drivers:
            await self.close(driver)
        await self.close(fresh)
        executable = Path(os.environ['LIVE_TEST_SCENE_DRIVER'])
        for i in range(len(plans)):
            graph = await offline(executable, '--usd-graph', directory / f'writer-{i}.usdc')
            write_json(directory / f'writer-{i}.graph.json', graph)
            self.assertEqual(self.export_public_fields(graph['host']), self.export_public_fields(retained[i]['host']))
        self.assertEqual(self.state.attempts, attempts, 'offline export reading performed network work')

    async def test_http2_cancel_reset_resumes_read_on_same_connection_through_tls(self):
        self.transport_tls = True
        await self.test_http2_cancel_reset_resumes_read_on_same_connection()

    async def test_http2_cancel_reset_resumes_read_on_same_connection(self):
        proxy = await self.http2_proxy(self.address)
        authority = await proxy.start()
        driver, initial = await self.start(authority, host=True)
        self.compare(initial)
        gate = self.state.plan('SubscribeReadLayer', 'before', attempt=2, pause=True)
        self.addCleanup(gate.release.set)
        faults = await proxy.reset_active(code=8)
        self.trace['http2_faults'] = faults
        await asyncio.wait_for(gate.reached.wait(), 5)
        # The service gate proves a new request was issued, but has supplied no
        # recovery coverage. Diagnostics must not announce it as resumed yet.
        waiting = (await driver.command('diagnostics'))['diagnostics']
        def events(snapshot, name):
            return [r for r in snapshot['recent_records'] if r['event_name'] == name and
                    r['subject_id'] == initial['subject']]
        ready = events(waiting, 'stream_resumed')
        self.assertEqual(len(ready), 1)
        first = ready[0]['fields']
        self.assertEqual((first['operation'], first['attempt_id'], first['stream_generation'],
                          first['resume_sequence'], first['read_sequence']), (3, 1, initial['generation'], 0, 1))
        self.assertGreater(first['request_id'], 0)
        for name in ('rpc_failed', 'stream_disconnected', 'retry_scheduled'):
            records = events(waiting, name)
            self.assertEqual(len(records), 1, (name, records))
            fields = records[0]['fields']
            self.assertEqual((fields['operation'], fields['result'], fields['request_id'],
                              fields['attempt_id'], fields['stream_generation']),
                             (3, 10, first['request_id'], 1, initial['generation']))
            if name == 'rpc_failed':
                self.assertEqual((fields['grpc_status'], fields['auth_generation']), (1, 1))
            else:
                self.assertEqual(fields['resume_sequence'], 0)
            if name == 'retry_scheduled':
                # The first reconnect waits below twice the driver's reconnect_initial_ms of 5 ms.
                self.assertLess(fields['delay_ms'], 10)
        sequence = await self.update(Delta(samples=(S(4, -0.0, 17), S(6, 2.0, 23))))
        gate.release.set()
        resumed = await driver.command(f'wait {sequence}')
        self.compare(resumed)
        self.assertEqual(resumed['subject'], initial['subject'])
        self.assertGreater(resumed['generation'], initial['generation'])
        self.assertEqual(self.state.reads[-1]['cursor'], 1)
        self.assertEqual(self.state.reads[-1]['backend'], 'A')
        self.assertEqual(len(proxy.connections), 1, 'stream reset unexpectedly replaced the TCP connection')
        submitted = await driver.command('sample 4 4611686018427387904 41')
        self.assertEqual(submitted['result'], 0, submitted)
        self.compare(await driver.command(f'wait {sequence + 1}'))
        self.assertEqual(decode(self.commits()[-1].payload).samples, (S(4, 2.0, 41),))
        self.assertEqual(self.commits()[-1].writer, initial['writer'])
        self.assertEqual(self.state.attempts['CreateLayer'], 1)
        self.assertEqual(self.state.attempts['UpdateLayer'], 2)
        completed = await driver.command('close-diagnostics')
        self.assertEqual(completed['result'], 0, completed)
        await asyncio.wait_for(driver.process.wait(), 8)
        await driver.drainer
        self.assertEqual(driver.process.returncode, 0, driver.log_tail())
        snapshot = completed['diagnostics']
        self.trace['read_diagnostics'] = snapshot
        ready = events(snapshot, 'stream_resumed')
        self.assertEqual(len(ready), 2, 'repeated coverage or close announced extra resumption')
        second = ready[1]['fields']
        self.assertEqual((second['operation'], second['attempt_id'], second['stream_generation'],
                          second['resume_sequence']), (3, 1, resumed['generation'], 1))
        self.assertNotEqual(second['request_id'], first['request_id'])
        self.assertEqual((second['read_sequence'], second['result']), (sequence, 0))
        self.assertEqual(len(events(snapshot, 'stream_disconnected')), 1, 'explicit close reported a disconnect')
        self.assertEqual(len(events(snapshot, 'retry_scheduled')), 1, 'explicit close scheduled a retry')
        self.assertEqual(len(events(snapshot, 'recovery_gate_closed')), 1)
        self.assertEqual(len(events(snapshot, 'recovery_gate_opened')), 1)
        failures = events(snapshot, 'rpc_failed')
        self.assertEqual([r['fields']['request_id'] for r in failures], [first['request_id'], second['request_id']])
        self.assertEqual([r['fields']['stream_generation'] for r in failures],
                         [initial['generation'], resumed['generation']])
        metrics = snapshot['metrics']
        self.assertEqual([m for m in metrics['rpc_attempts_by_operation'] if m['operation'] == 3],
                         [dict(operation=3, count=2)])
        self.assertEqual([m for m in metrics['rpc_by_operation_result'] if m['operation'] == 3],
                         [dict(operation=3, result=10, count=2)])
        self.assertEqual(snapshot['metrics']['rpc_in_flight'], 0)
        self.assertFalse(snapshot['history']['truncated'])
        for private in ('synthetic-writer', 'Bearer', 'root.live', authority, self.address, self.session, initial['writer']):
            self.assertNotIn(private, json.dumps(snapshot))
        self.trace['http2'] = proxy.snapshot()
        self.assertFalse(proxy.errors)

    async def test_http2_goaway_drains_read_and_routes_new_work_to_second_backend_through_tls(self):
        self.transport_tls = True
        await self.test_http2_goaway_drains_read_and_routes_new_work_to_second_backend()

    async def test_http2_goaway_drains_read_and_routes_new_work_to_second_backend(self):
        address_b, stub_b = await self.backend('B')
        proxy = await self.http2_proxy(self.address)
        driver, initial = await self.start(await proxy.start(), host=True)
        self.compare(initial)
        old = await proxy.drain_to(address_b)
        old_highest = [c.highest for c in old]
        # A PING ACK fences client receipt of GOAWAY. The existing read must
        # still deliver a peer edit; no wall-clock quiet period proves survival.
        sequence = await self.update(Delta(samples=(S(4, -0.0, 17),)))
        alive = await driver.command(f'wait {sequence}')
        self.compare(alive)
        self.assertEqual(alive['generation'], initial['generation'])
        self.assertEqual(alive['subject'], initial['subject'])
        self.assertEqual(self.state.attempts['SubscribeReadLayer'], 1)
        submitted = await driver.command('sample 6 4611686018427387904 41')
        self.assertEqual(submitted['result'], 0, submitted)
        self.compare(await driver.command(f'wait {sequence + 1}'))
        self.assertTrue(any(c.target == address_b and c.highest for c in proxy.connections))
        self.assertEqual([c.highest for c in old], old_highest, 'new stream opened on a drained connection')
        # End the old long-lived stream explicitly; the actor must then resume
        # against backend B using its preserved complete frontier.
        gate = self.state.plan('SubscribeReadLayer', 'before', attempt=2, pause=True)
        self.addCleanup(gate.release.set)
        self.trace['http2_faults'] = await proxy.reset_active(connections=old, code=8)
        await asyncio.wait_for(gate.reached.wait(), 5)
        peer = await self.update(Delta(samples=(S(4, 3.0, 99),)), stub_b)
        gate.release.set()
        resumed = await driver.command(f'wait {peer}')
        self.compare(resumed)
        self.assertEqual(resumed['subject'], initial['subject'])
        self.assertGreater(resumed['generation'], initial['generation'])
        self.assertEqual((self.state.reads[-1]['backend'], self.state.reads[-1]['cursor']), ('B', sequence + 1))
        self.assertEqual(decode(self.commits()[-2].payload).samples, (S(6, 2.0, 41),))
        self.assertEqual(self.commits()[-2].writer, initial['writer'])
        self.assertEqual(self.state.attempts['CreateLayer'], 1)
        self.assertEqual(self.state.attempts['UpdateLayer'], 3)
        fresh, opened = await self.start(address_b)
        self.compare(opened)
        await self.close(driver)
        await self.close(fresh)
        self.trace['http2'] = proxy.snapshot()
        self.assertFalse(proxy.errors)

    async def test_http2_reset_after_commit_retains_unknown_write_without_replay_through_tls(self):
        self.transport_tls = True
        await self.test_http2_reset_after_commit_retains_unknown_write_without_replay()

    async def test_http2_reset_after_commit_retains_unknown_write_without_replay(self):
        self.state.advertise_capabilities = False  # The unknown outcome is kept, never re-sent.
        proxy = await self.http2_proxy(self.address)
        # A typed host is needed for a stock-parseable USD export.
        base = await self.update(Delta(fields=(F(4, 'typeName', Token('int')), F(6, 'typeName', Token('int')))))
        driver, initial = await self.start(await proxy.start(), host=True)
        first = self.state.attempts['UpdateLayer'] + 1
        before = self.state.plan('UpdateLayer', 'before', attempt=first, pause=True)
        after = self.state.plan('UpdateLayer', 'after_commit', attempt=first, pause=True)
        read = self.state.plan_read_frame(self.state.reads[-1]['attempt'], base + 2, include_suppressed=True)
        for gate in (before, after, read):
            self.addCleanup(gate.release.set)
        head = await driver.command('sample 4 9223372036854775808 17')
        self.assertEqual(head['result'], 0, head)
        await asyncio.wait_for(before.reached.wait(), 5)
        queued = await driver.command('sample-map 4 4611686018427387904 23')
        self.assertEqual(queued['result'], 0, queued)
        original = queued['pending_intents']
        self.assertEqual([e['id'] for e in original], [head['accepted_id'], queued['accepted_id']])
        peer = await self.update(Delta(samples=(S(6, 3.0, 99),)))
        self.assertEqual(peer, base + 1)
        retained = await driver.command(f'journal-applied {peer}')
        self.assertEqual(retained['result'], 0, retained)
        self.assertEqual(retained['pending_intents'], original)
        self.assertEqual(retained['authoritative'], self.expected())
        model = Model()
        for commit in self.commits():
            model.apply(commit.payload, commit.sequence)
        for entry in original:
            model.apply(bytes.fromhex(entry['bytes']))
        self.assertEqual(retained['visible'], typed(model.state))
        # Applying the peer on the serial USD worker fences publication of the
        # rebased local suffix before commit/reply-loss can make it terminal.
        self.assertEqual(retained['host'], self.export_public_fields(retained['visible']))
        before.release.set()
        await asyncio.wait_for(asyncio.gather(after.reached.wait(), read.reached.wait()), 5)
        self.trace['http2_faults'] = await proxy.reset_active(code=8)
        await asyncio.wait_for(asyncio.gather(after.cancelled.wait(), read.cancelled.wait()), 5)
        failed = await driver.command(f'wait {peer + 1}')
        self.assertEqual((failed['result'], failed['status_result'], failed['status_state']), (19, 19, 7))
        self.assertEqual(failed['subject'], initial['subject'])
        self.assertGreater(failed['incident'], 0)
        for key in ('pending_intents', 'intent_provenance', 'authoritative', 'visible', 'host'):
            self.assertEqual(failed[key], retained[key])
        self.assertEqual(self.commits()[-1].payload.hex(), original[0]['bytes'])
        self.assertEqual(self.commits()[-1].writer, initial['writer'])
        self.assertEqual(decode(self.commits()[-1].payload).samples, (S(4, -0.0, 17),))
        self.assertEqual(len(self.commits()), peer + 1)
        diagnostics = (await driver.command('diagnostics'))['diagnostics']
        self.trace['uncertain_write_diagnostics'] = diagnostics
        writes = [r for r in diagnostics['recent_records'] if r['event_name'] == 'rpc_failed' and
                  r['fields']['operation'] == 4]
        self.assertEqual(len(writes), 1)
        write = writes[0]
        self.assertEqual(write['subject_id'], initial['subject'])
        fields = write['fields']
        self.assertEqual((fields['result'], fields['grpc_status'], fields['attempt_id'], fields['auth_generation'],
                          fields['stream_generation'], fields['work_id']),
                         (19, 1, 1, 1, initial['generation'], head['accepted_id']))
        self.assertGreater(fields['request_id'], 0)
        incidents = [r for r in diagnostics['recent_records'] if r['event_name'] == 'outcome_unknown']
        self.assertEqual(len(incidents), 1)
        self.assertEqual((incidents[0]['subject_id'], incidents[0]['incident_id']),
                         (initial['subject'], failed['incident']))
        statuses = [s for s in diagnostics['statuses'] if s['subject_id'] == initial['subject']]
        self.assertEqual(len(statuses), 1)
        self.assertEqual((statuses[0]['state'], statuses[0]['result'], statuses[0]['operation'],
                          statuses[0]['pending_write_count']), (7, 19, 4, 2))
        self.assertEqual([m for m in diagnostics['metrics']['rpc_by_operation_result'] if m['operation'] == 4],
                         [dict(operation=4, result=19, count=1)])
        self.assertEqual([m for m in diagnostics['metrics']['rpc_attempts_by_operation'] if m['operation'] == 4],
                         [dict(operation=4, count=1)])
        destination = self.artifact / (self.id().split('.')[-1] + '.usdc')
        # The driver's quoted argument treats a backslash as an escape; USD accepts forward slashes on Windows.
        exported = await driver.command(f'export "{destination.as_posix()}"')
        self.assertEqual(exported['result'], 0, exported)
        self.assertEqual(exported['pending_intents'], original)
        fresh, opened = await self.start()
        self.compare(opened)
        self.assertEqual(self.state.attempts['CreateLayer'], 1)
        self.assertEqual(self.state.attempts['UpdateLayer'], 3)
        await self.close(driver)
        await self.close(fresh)
        executable = Path(os.environ['LIVE_TEST_SCENE_DRIVER'])
        graph = await self.offline(executable, '--usd-graph', destination)
        self.assertEqual(self.export_public_fields(graph['host']), self.export_public_fields(retained['host']))
        self.assertEqual(self.state.attempts['UpdateLayer'], 3)
        self.trace['http2'] = proxy.snapshot()
        self.assertFalse(proxy.errors)

    async def test_http2_partial_payload_reset_preserves_cursor_and_queued_edit_through_tls(self):
        self.transport_tls = True
        await self.test_http2_partial_payload_reset_preserves_cursor_and_queued_edit()

    async def test_http2_partial_payload_reset_preserves_cursor_and_queued_edit(self):
        proxy = await self.http2_proxy(self.address)
        address_b, stub_b = await self.backend('B')
        driver, initial = await self.start(await proxy.start(), host=True)
        # The host applies the bootstrap in a job queued behind its open, so
        # the first report can still be OPENING; the later equality check
        # needs the applied bootstrap. The progress counter advances before
        # the READY status publishes, so only the applied count is pinned.
        initial = await driver.command(f"wait {initial['covered']}")
        self.assertEqual(initial['applied'], initial['covered'], initial)
        self.compare(initial)
        # The completed bootstrap is larger than a Ping reply. Require exactly
        # one still-open response with that known lower bound; no RPC-name or
        # HPACK guess is used to choose a stream.
        snapshot = self.state.sessions[self.session].layers['root.live'].snapshot_bytes
        streams = [(c, stream, size) for c in proxy.connections for stream, size in c.response_bytes.items()
                   if stream in c.active and size >= len(snapshot)]
        self.assertEqual(len(streams), 1, streams)
        connection, stream, baseline = streams[0]
        prefix = 7
        partial = self.state.plan_partial_read(self.state.reads[-1]['attempt'], 2, prefix)
        resumed_gate = self.state.plan('SubscribeReadLayer', 'before', attempt=2, pause=True)
        for gate in (partial, resumed_gate):
            self.addCleanup(gate.release.set)
        values = T(((-0.0, 17), (1.0, 23), (2.0, 31), (3.0, 41)))
        sequence = await self.update(Delta(fields=(F(4, 'timeSamples', values),)))
        self.assertEqual(sequence, 2)
        await asyncio.wait_for(partial.reached.wait(), 5)
        commit = self.commits()[-1]
        messages = [pb.SubscribeReadLayerResponse(header=pb.SubscribeReadLayerHeader(
            object_id=self.object, sequence_num=sequence, total_size=len(commit.payload),
            client_timestamp_microseconds_since_epoch=commit.client_timestamp_us,
            server_timestamp_microseconds_since_epoch=commit.timestamp_us)),
            pb.SubscribeReadLayerResponse(chunk=pb.SubscribeReadLayerChunk(data=commit.payload[:prefix]))]
        expected_bytes = baseline + sum(5 + message.ByteSize() for message in messages)
        observed = await asyncio.wait_for(proxy.wait_response_bytes(connection, stream, expected_bytes), 5)
        self.assertEqual(observed, expected_bytes)
        old = await proxy.drain_to(address_b)  # PING ACK follows the forwarded partial messages.
        incomplete = await driver.command('report')
        self.assertEqual(incomplete['result'], 0, incomplete)
        for key in ('subject', 'object', 'applied', 'covered', 'journal_applied', 'generation', 'authoritative', 'visible'):
            self.assertEqual(incomplete[key], initial[key])
        self.trace['partial_delivery'] = dict(connection=connection.id, stream=stream, baseline_bytes=baseline,
            forwarded_bytes=observed, prefix_bytes=prefix, complete_payload_bytes=len(commit.payload), sequence=sequence)
        self.trace['http2_faults'] = await proxy.reset_active(connections=old)
        self.assertIn(dict(action='reset', connection=connection.id, stream=stream, code=8), self.trace['http2_faults'])
        await asyncio.wait_for(asyncio.gather(partial.cancelled.wait(), resumed_gate.reached.wait()), 5)
        queued = await driver.command('sample 6 4611686018427387904 67')
        self.assertEqual(queued['result'], 0, queued)
        self.assertEqual([e['id'] for e in queued['pending_intents']], [queued['accepted_id']])
        self.assertEqual((queued['applied'], queued['covered'], queued['journal_applied']), (1, 1, 1))
        self.assertEqual(queued['authoritative'], initial['authoritative'])
        self.assertEqual(self.state.attempts['UpdateLayer'], 1, 'queued edit issued before read recovery')
        peer = await self.update(Delta(samples=(S(4, 4.0, 53),)), stub_b)
        self.assertEqual(peer, 3)
        resumed_gate.release.set()
        resumed = await driver.command(f'wait {peer + 1}')
        self.compare(resumed)
        self.assertEqual(resumed['subject'], initial['subject'])
        self.assertEqual(resumed['object'], initial['object'])
        self.assertGreater(resumed['generation'], initial['generation'])
        self.assertEqual((resumed['local_admitted'], resumed['local_retired']), (1, 1))
        self.assertEqual((self.state.reads[-1]['backend'], self.state.reads[-1]['cursor']), ('B', 1))
        self.assertEqual(self.state.attempts['SubscribeReadLayer'], 2)
        self.assertEqual(self.commits()[-1].payload.hex(), queued['pending_intents'][0]['bytes'])
        self.assertEqual(self.commits()[-1].writer, initial['writer'])
        self.assertEqual(decode(self.commits()[-1].payload).samples, (S(6, 2.0, 67),))
        value_node = next(n for n in resumed['visible'] if n['id'] == 4)
        self.assertEqual(value_node['fields']['timeSamples'], typed_value(T(values.value + ((4.0, 53),))))
        self.assertEqual(self.state.attempts['CreateLayer'], 1)
        self.assertEqual(self.state.attempts['UpdateLayer'], 3)
        fresh, opened = await self.start(address_b)
        self.compare(opened)
        await self.close(driver)
        await self.close(fresh)
        self.trace['http2'] = proxy.snapshot()
        self.assertFalse(proxy.errors)

    async def test_tls_proxy_that_rotates_to_an_untrusted_certificate_fails_the_layer_unavailable(self):
        self.transport_tls = True
        pki = await self.tls_pki()
        context = pki.server_context()
        proxy = RotatingProxy(self.address, ssl_context=context)
        self.proxies.append(proxy)
        driver, initial = await self.start(await proxy.start(), host=True)
        other = await self.tls_pki('other')
        context.load_cert_chain(other.certificate, other.key)  # New handshakes present an untrusted leaf.
        reads = self.state.attempts['SubscribeReadLayer']
        await proxy.rotate_and_wait(self.address)
        failed = await driver.command(f"wait {initial['covered'] + 1}")
        # Every reconnect fails verification; the runtime never falls back to plaintext.
        self.assertEqual((failed['result'], failed['status_state'], failed['status_result']), (8, 7, 8), failed)
        self.assertEqual(self.state.attempts['SubscribeReadLayer'], reads)
        await self.close(driver)
