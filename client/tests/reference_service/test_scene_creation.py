# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned explicit scene creation against an independent service ledger."""
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote
import unittest

import grpc
from service import Principal, ReferenceServer, pb, rpc
from graph_service import ERROR_DOMAIN, ErrorInfo, GraphState, GraphServer, RpcStatus
from scene_graph_wire import Command as C, Field as F, Delta, encode
from scene_graph_model import Model
from test_graph_transport import Driver, typed

AUTH = (("authorization", "Bearer synthetic-writer"),)


class SceneCreationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.artifact = Path(os.environ['LIVE_TEST_ARTIFACT_DIR'])
        self.state = GraphState(int(os.environ['LIVE_TEST_SEED']))
        self.state.tokens = {'synthetic-writer': Principal('writer')}
        self.drivers, self.channels, self.servers = [], [], []
        self.addAsyncCleanup(self.cleanup)
        self.address, self.stub = await self.backend(GraphServer)
        self.session = (await self.stub.CreateSession(
            pb.CreateSessionRequest(path='creation-base', name='creation'), metadata=AUTH, timeout=5)).session_id
        self.layer = 'created.live'
        self.source_format = 'usda'
        self.seed = encode(Delta(True, commands=(C(5, 1, spec_type=7), C(5, 2, 1, 'P', 6),
                                                C(5, 3, 2, 'Child', 6)),
                                 fields=(F(2, 'counter', 17), F(3, 'weight', -2.5)),
                                 source_format=self.source_format))
        model = Model()
        model.apply(self.seed, 1)
        self.expected = typed(model.state)
        self.seed_path = self.artifact / (self.id().split('.')[-1] + '.creation-seed.bin')
        self.seed_path.write_bytes(self.seed)

    def ordinary_seed(self):
        # Preserve coverage of the legal ordinary sequence-one bootstrap.
        # Ordinary deltas cannot carry a nonempty sourceFormat.
        self.source_format = ''
        self.seed = encode(Delta(commands=(C(1, 2, 1, 'P', 6), C(1, 3, 2, 'Child', 6)),
                                 fields=(F(2, 'counter', 17), F(3, 'weight', -2.5))))
        model = Model()
        model.apply(self.seed, 1)
        self.expected = typed(model.state)
        self.seed_path.write_bytes(self.seed)

    async def backend(self, server_type):
        server = server_type(self.state)
        self.servers.append(server)
        address = await server.start()
        channel = grpc.aio.insecure_channel(address)
        self.channels.append(channel)
        await asyncio.wait_for(channel.channel_ready(), 5)
        return address, rpc.LiveServiceStub(channel)

    async def cleanup(self):
        for fault in self.state.faults:
            fault.release.set()
        for driver in self.drivers:
            if driver.process.returncode is None:
                try:
                    driver.process.kill()
                except ProcessLookupError:
                    pass
        failures = []
        for jobs in ((d.process.wait() for d in self.drivers), (d.drainer for d in self.drivers),
                     (c.close() for c in self.channels), (s.stop() for s in self.servers)):
            try:
                values = await asyncio.wait_for(asyncio.gather(*jobs, return_exceptions=True), 5)
                failures.extend(repr(value) for value in values if isinstance(value, BaseException))
            except Exception as error:
                failures.append(repr(error))
        report = dict(seed=int(os.environ['LIVE_TEST_SEED']), ledger=self.state.snapshot(),
                      attempts=self.state.attempts, reads=self.state.reads, cleanup_errors=failures)
        (self.artifact / (self.id().split('.')[-1] + '.creation.json')).write_text(json.dumps(report, indent=2) + '\n')
        self.state.assert_faults_fired()
        self.assertFalse(failures)
        self.assertTrue(all(driver.error is None for driver in self.drivers))

    async def prepare(self, address=None, host=False, target=False, diagnostics=False):
        attempts = dict(self.state.attempts)
        process = await asyncio.create_subprocess_exec(
            os.environ['LIVE_TEST_SCENE_DRIVER'], '--create-target' if target else ('--create-host' if host else '--create'),
            address or self.address,
            self.session, self.layer, str(self.seed_path), *(['--diagnostics'] if diagnostics else []), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        driver = Driver(process, self.artifact / (self.id().split('.')[-1] + '.creation-native.log'))
        self.drivers.append(driver)
        prepared = await driver.receive()
        self.assertEqual(prepared['result'], 0, prepared)
        self.subject = prepared['subject']
        self.initial_revision = prepared['revision']
        self.assertGreater(self.subject, 0)
        self.assertGreater(self.initial_revision, 0)
        self.assertEqual((prepared['status_state'], prepared['status_result'], prepared['status_pending'],
                          prepared['status_bytes'], prepared['status_progress_known']), (1, 0, 0, 0, 0))
        self.assertEqual(prepared['phase'], 'prepared')
        self.assertFalse(prepared['started'])
        self.assertEqual((prepared['object'], prepared['sequence'], prepared['request']), (0, 0, 0))
        self.retained(prepared)
        self.assertEqual(self.state.attempts, attempts, 'prepare must not issue service RPCs')
        if target:
            self.assertTrue(prepared['target_mode'])
            self.assertTrue(prepared['target_alive'])
            self.assertFalse(prepared['layer_exposed'])
            canonical = 'live://' + self.address + '/' + quote(self.session, safe='') + '/' + quote(self.layer, safe='')
            self.assertEqual(prepared['private_host']['identifier'], canonical)
            self.assertEqual(prepared['private_host']['resolved_path'], canonical)
            expected = copy.deepcopy(self.expected)
            next(node for node in expected if node['path'] == '/P')['fields']['counter']['value'] = 5
            self.assertEqual(prepared['private_host']['scene'], expected)
            self.assertEqual(prepared['private_host']['source_format'], self.source_format)
        elif host:
            self.assertTrue(prepared['host_mode'])
            self.assertFalse(prepared['layer_exposed'])
            self.assertFalse(prepared['registered'])
            self.assertEqual(prepared['identifier_notices'], 0)
            self.assertEqual(prepared['host'], [])
            self.assertTrue(prepared['private_host']['identifier'].startswith('anon:'))
            self.assertEqual(prepared['private_host']['resolved_path'], '')
            self.assertEqual(prepared['private_host']['scene'], typed(Model().state))
            self.assertEqual(prepared['private_host']['source_format'], '')
        await self.observed_status(driver, 1, 0)
        return driver, prepared

    def retained(self, report, request=None):
        self.assertEqual(report['subject'], self.subject)
        self.assertGreaterEqual(report['revision'], self.initial_revision)
        self.assertEqual(report['seed'], self.expected)
        self.assertEqual(report['source_formats']['seed'], self.source_format)
        if report['journal_applied']:
            self.assertEqual(report['source_formats']['authoritative'], self.source_format)
            self.assertEqual(report['source_formats']['visible'], self.source_format)
        if report['layer_exposed']:
            self.assertEqual(report['private_host']['source_format'], self.source_format)
        self.assertEqual(bytes.fromhex(report['bytes']), self.seed)
        if request is not None:
            self.assertGreater(request, 0)
            self.assertEqual(report['request'], request)

    def commits(self):
        return [entry for entry in self.state.ledger
                if entry.session == self.session and entry.layer == self.layer]

    def committed_once(self, object_id=None):
        entries = self.commits()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].operation, 'CreateLayer')
        self.assertEqual(entries[0].payload, self.seed)
        self.assertEqual(entries[0].sequence, 1)
        if object_id is not None:
            self.assertEqual(entries[0].object_id, object_id)
        self.assertEqual(self.state.attempts.get('CreateLayer', 0), 1)
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), 0)

    async def confirmed(self, driver):
        started = await driver.command('start')
        self.assertEqual(started['result'], 0, started)
        self.assertTrue(started['started'])
        self.assertGreater(started['request'], 0)
        self.retained(started)
        ready = await driver.command('wait 5000')
        self.assertEqual(ready['result'], 0, ready)
        self.assertEqual(ready['create_result'], 0)
        self.assertEqual(ready['phase'], 'confirmed')
        self.assertTrue(ready['quiescent'])
        self.assertGreater(ready['object'], 0)
        self.assertEqual(ready['sequence'], 1)
        self.retained(ready, started['request'])
        self.assertEqual((ready['status_state'], ready['status_result'], ready['status_pending'],
                          ready['status_bytes'], ready['status_progress_known']), (1, 0, 0, 0, 0))
        self.committed_once(ready['object'])
        return ready

    async def test_retained_hook_unwind_preserves_unknown_creation(self):
        driver, _ = await self.prepare(host=True)
        attempts = dict(self.state.attempts)
        retained = await driver.command('registry-retain')
        self.assertEqual(retained['result'], 0)
        self.assertFalse(retained['started'])
        self.assertEqual(self.state.attempts, attempts)
        fault = self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await driver.command('registry-start'))['result'], 0)
        await asyncio.wait_for(fault.reached.wait(), 5)
        done = await driver.command('wait 5000')
        self.assertEqual((done['phase'], done['quiescent']), ('unknown', True))
        self.committed_once()
        await driver.command('replace-host')
        taken = await driver.command('registry-take')
        self.assertEqual(taken['subject'], self.subject)
        self.assertEqual(taken['request'], done['request'])
        self.assertEqual((await driver.command('start'))['result'], 5)
        self.committed_once()
        await self.quit(driver)

    async def test_hook_failure_before_issuance_preserves_unissued_result_and_seed(self):
        driver, _ = await self.prepare(host=True)
        attempts = dict(self.state.attempts)
        self.assertEqual((await driver.command('registry-retain'))['result'], 0)
        self.assertEqual((await driver.command('hook-fail-timeout'))['result'], 0)
        failed = await driver.command('wait 5000')
        self.assertEqual((failed['phase'], failed['create_result'], failed['request'], failed['quiescent']),
                         ('unissued', 9, 0, True))
        self.assertEqual((failed['status_state'], failed['status_result'], failed['status_pending'],
                          failed['status_unconfirmed'], failed['status_grpc_present']), (7, 9, 0, 0, 0))
        self.assertEqual((await driver.command('start'))['result'], 5)
        self.assertEqual(self.state.attempts, attempts)
        await driver.command('registry-take')
        await self.quit(driver)

    async def abandoned_hook(self, boundary, code, phase, result):
        driver, _ = await self.prepare(host=True)
        self.assertEqual((await driver.command('registry-retain'))['result'], 0)
        fault = self.state.plan('CreateLayer', boundary, pause=True, code=code)
        self.assertEqual((await driver.command('registry-start'))['result'], 0)
        await asyncio.wait_for(fault.reached.wait(), 5)
        pending = await driver.command('hook-fail-timeout')
        self.assertEqual((pending['result'], pending['phase'], pending['status_state'], pending['status_result']),
                         (0, 'creating', 1, 9))
        self.assertEqual((pending['incident'], pending['status_pending'], pending['status_unconfirmed'],
                          pending['status_grpc_present']), (0, 1, 1, 0))
        # Inject the failure at an observed service boundary, not by waiting
        # out a deadline. The original finite operation is still allowed to
        # settle; it is neither cancelled nor retried by the failed hook.
        fault.release.set()
        failed = await driver.command('wait 5000')
        self.assertEqual((failed['result'], failed['phase'], failed['create_result'], failed['quiescent']),
                         (0, phase, result, True))
        unknown = phase == 'unknown'
        self.assertEqual((failed['status_pending'], failed['status_unconfirmed'], failed['status_progress_known']),
                         (int(unknown), int(unknown), 0))
        self.assertEqual(failed['status_operation'], 1 if phase == 'confirmed' else 2)
        self.assertEqual(failed['status_result'], 9 if phase == 'confirmed' else result)
        self.assertGreater(failed['incident'], 0)
        if phase == 'confirmed':
            self.assertEqual((failed['status_grpc_present'], failed['status_grpc']), (0, 0))
        else:
            self.assertEqual((failed['status_grpc_present'], failed['status_grpc']), (1, code.value[0]))
        await driver.command('registry-take')
        self.assertEqual((await driver.command('host-prepare 0'))['result'], 9)
        self.assertEqual((await driver.command('activate'))['result'], 9)
        self.assertEqual((await driver.command('start'))['result'], 5)
        self.assertEqual(self.state.attempts.get('CreateLayer'), 1)
        self.assertEqual(self.state.attempts.get('SubscribeReadLayer', 0), 0)
        await self.quit(driver)
        return failed

    async def test_abandoned_hook_late_confirmation_preserves_local_open_failure(self):
        failed = await self.abandoned_hook('after_commit', None, 'confirmed', 0)
        self.committed_once(failed['object'])

    async def test_abandoned_hook_permission_error_is_a_definitive_rejection(self):
        # The service checks write permission before commit (design section 13), so
        # the client classifies the rejection from its status, not the ledger.
        await self.abandoned_hook('before', grpc.StatusCode.PERMISSION_DENIED, 'rejected', 7)
        self.assertFalse(self.commits())

    async def test_abandoned_hook_lost_reply_preserves_uncertainty_and_pending_seed(self):
        await self.abandoned_hook('after_commit', grpc.StatusCode.UNAVAILABLE, 'unknown', 19)
        self.committed_once()

    async def test_retained_cleanup_keeps_slot_busy_through_final_host_destruction(self):
        driver, _ = await self.prepare(host=True)
        attempts = dict(self.state.attempts)
        self.assertEqual((await driver.command('registry-retain'))['result'], 0)
        closed = await driver.command('registry-close')
        self.assertEqual((closed['result'], closed['phase'], closed['quiescent']), (0, 'unissued', True))
        self.assertEqual(self.state.attempts, attempts)
        self.assertFalse(self.commits())
        await self.quit(driver)

    async def test_retained_busy_cleanup_preserves_queued_callback_until_explicit_release(self):
        driver, _ = await self.prepare(host=True)
        self.assertEqual((await driver.command('registry-retain'))['result'], 0)
        receipt = await self.confirmed(driver)
        self.assertEqual((await driver.command('registry-host-prepare 5000'))['result'], 0)
        activated = await driver.command('registry-activate')
        self.assertEqual((activated['result'], activated['registered']), (0, True))
        payload = encode(Delta(fields=(F(3, 'weight', 91.5),)))
        frames = [pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name=self.layer,
                    object_id=receipt['object'], client_id='retained-cleanup-peer')),
                  pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=payload))]
        sequence = (await self.stub.UpdateLayer(iter(frames), metadata=AUTH, timeout=5)).sequence_num
        self.assertEqual(sequence, 2)
        # Native CV waits for the queued event. It asserts BUSY while the
        # accepted closure is held, then releases it and awaits quiescence.
        closed = await driver.command('registry-cancel-held')
        self.assertEqual(closed['result'], 0)
        self.assertFalse(closed['registered'])
        self.assertEqual([entry.payload for entry in self.commits()], [self.seed, payload])
        self.assertEqual(self.state.attempts.get('CreateLayer'), 1)
        self.assertEqual(self.state.attempts.get('UpdateLayer'), 1)
        await self.quit(driver)

    async def observed_status(self, driver, state, result):
        report = await driver.command('observed-status')
        self.assertEqual(report['result'], 0, report)
        self.assertEqual((report['status_state'], report['status_result']), (state, result))
        self.assertEqual((report['observed_subject'], report['observed_state'], report['observed_result']),
                         (self.subject, state, result))
        self.assertEqual(report['observed_incident'], report['incident'])
        self.assertEqual(report['observed_revision'], report['revision'])
        self.retained(report)
        return report

    async def terminal_status(self, driver, result, state=7):
        failed = await self.host_until(driver, lambda report:
                                       report['status_state'] == state and report['status_result'] == result)
        self.assertGreater(failed['incident'], 0)
        await self.observed_status(driver, state, result)
        self.retained(failed)
        return failed

    async def quit(self, driver):
        report = await driver.command('quit')
        self.assertEqual(report['result'], 0, report)
        self.retained(report)
        await asyncio.wait_for(driver.process.wait(), 8)
        await driver.drainer
        self.assertEqual(driver.process.returncode, 0, driver.path.read_text()[-8192:])
        self.assertTrue(all(d.process.returncode == 0 for d in self.drivers))

    async def test_creation_prepared_reservation_single_attempt_and_confirmed_attach(self):
        driver, _ = await self.prepare()
        self.assertEqual((await driver.command('wait 1'))['result'], 14)
        self.assertEqual((await driver.command('attach'))['result'], 14)
        duplicate = await driver.command('duplicate')
        self.assertEqual(duplicate['result'], 5, duplicate)
        self.retained(duplicate)
        self.assertEqual(self.state.attempts.get('CreateLayer', 0), 0)
        ready = await self.confirmed(driver)
        repeated = await driver.command('start')
        self.assertEqual(repeated['result'], 5)
        self.retained(repeated, ready['request'])
        attached = await driver.command('attach')
        self.assertEqual(attached['result'], 0, attached)
        self.assertTrue(attached['attached'])
        self.assertEqual(attached['attach_result'], 0)
        self.assertEqual(attached['attach_object'], ready['object'])
        self.assertEqual(attached['applied'], 1)
        self.assertEqual(attached['authoritative'], self.expected)
        self.assertEqual(attached['visible'], self.expected)
        self.retained(attached, ready['request'])
        self.assertEqual(self.state.reads, [dict(backend='A', attempt=1, object=ready['object'], cursor=0)])
        self.assertEqual((await driver.command('attach'))['result'], 0)
        self.assertEqual(self.state.attempts.get('SubscribeReadLayer'), 1)
        await self.observed_status(driver, 2, 0)
        closed_creation = await driver.command('close 5000')
        self.assertEqual(closed_creation['result'], 0)
        self.assertTrue(closed_creation['closing'])
        self.assertEqual((closed_creation['status_state'], closed_creation['status_result']), (2, 0))
        self.retained(closed_creation, ready['request'])
        self.committed_once(ready['object'])
        await self.quit(driver)

    async def proven_creation(self, command=None, *, unknown=False):
        """Starts a creation on an endpoint marked as supporting operation IDs, with its first reply lost.

        The creation is the driver endpoint's first write, so the driver marks the endpoint as an earlier reply
        reporting an operation outcome would."""
        driver, _ = await self.prepare(host=True)
        self.assertEqual((await driver.command('mark-operation-ids'))['result'], 0)
        if command:
            self.assertEqual((await driver.command(command))['result'], 0)
        plan = self.state.plan_unknown if unknown else self.state.plan
        plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        started = await driver.command('start')
        self.assertEqual(started['result'], 0)
        done = await driver.command('wait 5000')
        self.assertEqual((done['result'], done['quiescent']), (0, True), done)
        self.retained(done, started['request'])
        return driver, done

    async def test_creation_lost_reply_on_a_proven_endpoint_is_resent_and_confirmed(self):
        driver, done = await self.proven_creation()
        # The re-send earns the original object with ALREADY_COMMITTED; the service holds one creation.
        self.assertEqual((done['phase'], done['create_result']), ('confirmed', 0), done)
        self.assertEqual((done['object'], done['sequence']), (self.commits()[0].object_id, 1))
        self.assertEqual([entry['outcome'] for entry in self.state.operation_log], ['APPLIED', 'ALREADY_COMMITTED'])
        self.assertEqual([c.operation for c in self.commits()], ['CreateLayer'])
        self.assertEqual(self.state.attempts['CreateLayer'], 2)
        await self.quit(driver)

    async def test_creation_resend_whose_reply_is_lost_too_stays_unknown(self):
        driver, done = await self.proven_creation(unknown=True)
        self.assertEqual((done['phase'], done['create_result'], done['object']), ('unknown', 19, 0), done)
        await self.observed_status(driver, 7, 19)
        self.assertEqual((await driver.command('start'))['result'], 5)
        self.assertEqual(self.state.attempts['CreateLayer'], 2)  # No third attempt.
        await self.quit(driver)
        self.assertEqual([c.operation for c in self.commits()], ['CreateLayer'])

    async def test_creation_closed_while_its_resend_is_queued_issues_no_second_create(self):
        driver, done = await self.proven_creation('resend-close')
        self.assertEqual((done['phase'], done['create_result']), ('unknown', 19), done)
        self.assertTrue(done['closing'])
        self.assertEqual(self.state.attempts['CreateLayer'], 1)
        await self.quit(driver)

    async def test_creation_resend_that_cannot_be_submitted_stays_unknown(self):
        driver, done = await self.proven_creation('resend-fail')
        self.assertEqual((done['phase'], done['create_result']), ('unknown', 19), done)
        await self.observed_status(driver, 7, 19)
        self.assertEqual(self.state.attempts['CreateLayer'], 1)
        await self.quit(driver)

    def creation_records(self, diagnostics, name):
        """The creation's records of one event in a diagnostics snapshot."""
        return [r for r in diagnostics['recent_records'] if r['event_name'] == name and r['fields']['operation'] == 2]

    async def test_creation_refused_with_nothing_applied_is_reissued_after_the_retry_delay(self):
        # The service refuses the first CreateLayer UNAVAILABLE with its ErrorInfo and a RetryInfo of 15 ms: nothing
        # applied, so the creation is issued once more, with the same seed bytes and operation ID, no sooner than that
        # delay, and its layer activates READY.
        driver, _ = await self.prepare(host=True, diagnostics=True)
        self.state.refuse_creates(1, retry_delay_ms=15)
        started = await driver.command('start')
        self.assertEqual(started['result'], 0, started)
        receipt = await driver.command('wait 5000')
        self.assertEqual((receipt['result'], receipt['phase'], receipt['create_result'], receipt['quiescent'],
                          receipt['sequence']), (0, 'confirmed', 0, True, 1), receipt)
        self.retained(receipt, started['request'])
        self.assertEqual([(c.operation, c.payload, c.object_id) for c in self.commits()],
                         [('CreateLayer', self.seed, receipt['object'])])
        self.assertEqual(self.state.attempts['CreateLayer'], 2)  # The refusal and the reissue; nothing more.
        refused, creates = self.state.refusals, self.state.creates
        self.assertEqual([(r['method'], r['attempt']) for r in refused], [('CreateLayer', 1)])
        self.assertTrue(refused[0]['operation_id'])
        digest = hashlib.sha256(self.seed).hexdigest()
        self.assertEqual([(c['attempt'], c['operation_id'], c['sha256']) for c in creates],
                         [(1, refused[0]['operation_id'], digest), (2, refused[0]['operation_id'], digest)])
        self.assertEqual([entry['outcome'] for entry in self.state.operation_log], ['APPLIED'])
        # The service's retry delay is the least wait before the reissue.
        self.assertGreaterEqual(creates[1]['monotonic_ns'] - refused[0]['monotonic_ns'], 15_000_000)
        diagnostics = (await driver.command('diagnostics'))['diagnostics']
        failed = self.creation_records(diagnostics, 'rpc_failed')
        self.assertEqual([(r['fields']['result'], r['fields']['grpc_status'], r['fields']['request_id'])
                          for r in failed], [(8, 14, started['request'])])
        retries = self.creation_records(diagnostics, 'retry_scheduled')
        self.assertEqual([(r['subject_id'], r['fields']['result'], r['fields']['request_id'], r['fields']['attempt_id'],
                           r['fields']['reason'], r['fields']['work_id']) for r in retries],
                         [(self.subject, 8, started['request'], 1, 1, 0)])
        # The RetryInfo delay, or the reconnect draw when longer, within the driver's reconnect_maximum_ms of 20 ms.
        self.assertTrue(15 <= retries[0]['fields']['delay_ms'] <= 20, retries[0])
        await self.host_staged(driver, receipt)
        await self.host_activated(driver, receipt)
        self.assertEqual(self.state.attempts.get('SubscribeReadLayer'), 1)
        await self.quit(driver)

    async def test_creation_refusals_with_nothing_applied_fail_unavailable_after_the_attempt_bound(self):
        # The creation driver keeps the default reconnect_max_attempts of 10: the tenth consecutive refusal with nothing
        # applied settles the creation UNAVAILABLE, a definitive rejection, with its seed kept. A RetryInfo delay above
        # reconnect_maximum_ms waits that maximum, 20 ms.
        driver, _ = await self.prepare(host=True, diagnostics=True)
        self.state.refuse_creates(10, retry_delay_ms=1000)
        started = await driver.command('start')
        self.assertEqual(started['result'], 0, started)
        failed = await driver.command('wait 5000')
        self.assertEqual((failed['result'], failed['phase'], failed['create_result'], failed['quiescent'],
                          failed['object']), (0, 'rejected', 8, True, 0), failed)
        self.retained(failed, started['request'])
        observed = await self.observed_status(driver, 7, 8)  # FAILED, UNAVAILABLE
        self.assertEqual((observed['status_grpc_present'], observed['status_grpc']), (1, 14), observed)
        self.assertEqual(self.commits(), [])
        self.assertEqual((self.state.attempts['CreateLayer'], len(self.state.refusals)), (10, 10))
        self.assertEqual({(c['operation_id'], c['sha256']) for c in self.state.creates},
                         {(self.state.refusals[0]['operation_id'], hashlib.sha256(self.seed).hexdigest())})
        diagnostics = (await driver.command('diagnostics'))['diagnostics']
        self.assertEqual([r['fields']['result'] for r in self.creation_records(diagnostics, 'rpc_failed')], [8] * 10)
        retries = self.creation_records(diagnostics, 'retry_scheduled')
        self.assertEqual([(r['fields']['attempt_id'], r['fields']['reason'], r['fields']['delay_ms']) for r in retries],
                         [(attempt, 1, 20) for attempt in range(1, 10)])
        self.assertEqual((await driver.command('start'))['result'], 5)
        self.assertEqual(self.state.attempts.get('SubscribeReadLayer', 0), 0)
        await self.quit(driver)

    async def refused_once(self, *commands, retry_delay_ms=15):
        """Starts a creation the service refuses once with nothing applied, after the driver `commands`, and
        returns its settled receipt and diagnostics; no reissue reaches the service."""
        driver, _ = await self.prepare(host=True, diagnostics=True)
        for command in commands:
            self.assertEqual((await driver.command(command))['result'], 0, command)
        self.state.refuse_creates(1, retry_delay_ms=retry_delay_ms)
        started = await driver.command('start')
        self.assertEqual(started['result'], 0, started)
        failed = await driver.command('wait 5000')
        self.assertEqual((failed['result'], failed['phase'], failed['create_result'], failed['quiescent'],
                          failed['object']), (0, 'rejected', 8, True, 0), failed)
        self.retained(failed, started['request'])
        self.assertEqual(self.state.attempts['CreateLayer'], 1)  # The refused attempt; no reissue.
        self.assertEqual(len(self.state.refusals), 1)
        self.assertEqual(self.commits(), [])
        diagnostics = (await driver.command('diagnostics'))['diagnostics']
        return driver, failed, diagnostics

    async def test_creation_reissue_whose_backoff_ends_past_the_callers_deadline_is_not_issued(self):
        # The caller's initial-open deadline, 10 ms after it is set and so before the refusal arrives plus its
        # RetryInfo delay of 15 ms, bounds the reissue instead of initial_open_timeout_ms from start (4 s here), as an
        # Export or first Save hook's deadline does: the refusal stands as UNAVAILABLE and nothing is reissued.
        driver, _, diagnostics = await self.refused_once('reissue-deadline 10')
        await self.observed_status(driver, 7, 8)
        self.assertEqual(self.creation_records(diagnostics, 'retry_scheduled'), [])
        await self.quit(driver)

    async def test_creation_reissue_queued_past_its_deadline_is_not_issued(self):
        # The backoff ends before the deadline, but the dispatch worker runs the queued reissue after it: the worker
        # rechecks the deadline before it submits, so the refusal stands as UNAVAILABLE.
        driver, _, diagnostics = await self.refused_once('reissue-expire')
        await self.observed_status(driver, 7, 8)
        self.assertEqual([r['fields']['attempt_id'] for r in self.creation_records(diagnostics, 'retry_scheduled')],
                         [1])
        await self.quit(driver)

    async def test_creation_closed_with_its_reissue_queued_issues_no_reissue(self):
        # The creation closes while its reissue waits for the dispatch worker: the refusal stands as UNAVAILABLE.
        driver, failed, diagnostics = await self.refused_once('resend-close')
        self.assertTrue(failed['closing'])
        self.assertEqual([r['fields']['attempt_id'] for r in self.creation_records(diagnostics, 'retry_scheduled')],
                         [1])
        await self.quit(driver)

    async def test_creation_whose_hook_abandoned_the_opening_issues_no_reissue(self):
        # The hook fails with TIMEOUT while the attempt is in flight, abandoning the opening; the service then refuses
        # the attempt with nothing applied. No reissue follows the abandoned opening.
        driver, _ = await self.prepare(host=True, diagnostics=True)
        self.assertEqual((await driver.command('registry-retain'))['result'], 0)
        fault = self.state.plan('CreateLayer', 'before', pause=True)
        self.state.refuse_creates(1, retry_delay_ms=15)
        self.assertEqual((await driver.command('registry-start'))['result'], 0)
        await asyncio.wait_for(fault.reached.wait(), 5)
        pending = await driver.command('hook-fail-timeout')
        self.assertEqual((pending['result'], pending['phase'], pending['status_result']), (0, 'creating', 9), pending)
        fault.release.set()
        failed = await driver.command('wait 5000')
        self.assertEqual((failed['result'], failed['phase'], failed['create_result'], failed['quiescent']),
                         (0, 'rejected', 8, True), failed)
        self.assertEqual((self.state.attempts['CreateLayer'], len(self.state.refusals)), (1, 1))
        diagnostics = (await driver.command('diagnostics'))['diagnostics']
        self.assertEqual(self.creation_records(diagnostics, 'retry_scheduled'), [])
        await driver.command('registry-take')
        await self.quit(driver)
        self.assertEqual(self.commits(), [])

    async def test_creation_refused_without_the_service_error_info_is_not_reissued(self):
        # A plain UNAVAILABLE, one with only a RetryInfo, one whose ErrorInfo names another domain and an INTERNAL with
        # the service's ErrorInfo do not prove that nothing applied: the creation's outcome is unknown, and on an
        # endpoint that has not shown operation-ID support it is neither reissued nor re-sent.
        internal = RpcStatus(code=grpc.StatusCode.INTERNAL.value[0], message='internal')
        internal.details.add(type_url='type.googleapis.com/google.rpc.ErrorInfo',
                             value=ErrorInfo(reason='INTERNAL', domain=ERROR_DOMAIN).SerializeToString())
        cases = (dict(error_info=False), dict(error_info=False, retry_delay_ms=5), dict(domain='other.domain'), None)
        for case in cases:
            with self.subTest(case=case):
                driver, _ = await self.prepare(host=True, diagnostics=True)
                creates = self.state.attempts.get('CreateLayer', 0)
                if case is None:
                    self.state.plan('CreateLayer', 'before', code=grpc.StatusCode.INTERNAL,
                                    details=internal.SerializeToString())
                else:
                    self.state.refuse_creates(1, **case)
                started = await driver.command('start')
                self.assertEqual(started['result'], 0, started)
                failed = await driver.command('wait 5000')
                self.assertEqual((failed['result'], failed['phase'], failed['create_result']), (0, 'unknown', 19), failed)
                self.retained(failed, started['request'])
                await self.observed_status(driver, 7, 19)
                self.assertEqual(self.state.attempts['CreateLayer'], creates + 1)
                diagnostics = (await driver.command('diagnostics'))['diagnostics']
                self.assertEqual(self.creation_records(diagnostics, 'retry_scheduled'), [])
                self.assertEqual([r['fields']['result'] for r in self.creation_records(diagnostics, 'rpc_failed')],
                                 [19])
                await self.quit(driver)
        self.assertEqual(self.commits(), [])

    async def test_creation_lost_reply_retains_seed_receipt_without_resend(self):
        driver, _ = await self.prepare(host=True)
        fault = self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        started = await driver.command('start')
        self.assertEqual(started['result'], 0)
        await asyncio.wait_for(fault.reached.wait(), 5)
        failed = await driver.command('wait 5000')
        self.assertEqual(failed['result'], 0, failed)
        self.assertEqual((failed['phase'], failed['create_result']), ('unknown', 19))
        self.assertEqual((failed['object'], failed['sequence']), (0, 0))
        self.assertTrue(failed['quiescent'])
        self.retained(failed, started['request'])
        observed = await self.observed_status(driver, 7, 19)
        self.assertGreater(observed['incident'], 0)
        self.assertEqual((observed['status_pending'], observed['status_bytes'], observed['status_progress_known']),
                         (1, len(self.seed), 0))
        diagnostic = await driver.command('diagnostics')
        self.assertEqual(diagnostic['diagnostics']['recent_records'], [])
        self.assertEqual((diagnostic['subject'], diagnostic['status_result'], diagnostic['incident']),
                         (self.subject, 19, observed['incident']))
        self.assertEqual((await driver.command('start'))['result'], 5)
        refused = await driver.command('attach')
        self.assertEqual(refused['result'], 19, refused)
        self.assertFalse(refused['attached'])
        self.retained(refused, started['request'])
        self.assertEqual((await driver.command('duplicate'))['result'], 5)
        self.assertEqual(self.state.attempts.get('SubscribeReadLayer', 0), 0)
        self.committed_once()
        await self.quit(driver)
        self.committed_once()

    async def test_creation_unknown_diagnostic_correlates_with_retained_seed(self):
        driver, _ = await self.prepare(host=True, diagnostics=True)
        fault = self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        started = await driver.command('start')
        self.assertEqual(started['result'], 0)
        await asyncio.wait_for(fault.reached.wait(), 5)
        failed = await driver.command('wait 5000')
        self.assertEqual((failed['result'], failed['create_result'], failed['quiescent']), (0, 19, True))
        self.retained(failed, started['request'])
        observed = await self.observed_status(driver, 7, 19)
        self.assertGreater(observed['incident'], 0)
        snapshot = (await driver.command('diagnostics'))['diagnostics']
        records = [r for r in snapshot['recent_records'] if r['event_name'] == 'rpc_failed']
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record['subject_id'], self.subject)
        fields = record['fields']
        self.assertEqual((fields['operation'], fields['result'], fields['request_id'], fields['attempt_id'],
                          fields['auth_generation'], fields['grpc_status'], fields['stream_generation'], fields['work_id']),
                         (2, 19, started['request'], 1, 1, 14, 0, 0))
        incidents = [r for r in snapshot['recent_records'] if r['event_name'] == 'outcome_unknown']
        self.assertEqual(len(incidents), 1)
        self.assertEqual((incidents[0]['subject_id'], incidents[0]['incident_id']),
                         (self.subject, observed['incident']))
        self.assertEqual(snapshot['metrics']['rpc_attempts_by_operation'], [dict(operation=2, count=1)])
        self.assertEqual(snapshot['metrics']['rpc_by_operation_result'], [dict(operation=2, result=19, count=1)])
        for private in ('synthetic-writer', 'Bearer', self.address, self.session, self.layer, self.seed.hex()):
            self.assertNotIn(private, json.dumps(snapshot))
        self.committed_once()
        self.assertEqual(self.state.attempts.get('SubscribeReadLayer', 0), 0)
        await self.quit(driver)

    async def test_creation_over_the_service_entry_limit_reports_its_counts(self):
        # The seed's three sections and two fields are five entries; this
        # service holds four, while the client counts against the default.
        self.state.delta_entry_limit = 4
        driver, _ = await self.prepare(host=True, diagnostics=True)
        started = await driver.command('start')
        self.assertEqual(started['result'], 0)
        rejected = await driver.command('wait 5000')
        self.assertEqual((rejected['phase'], rejected['create_result'], rejected['quiescent']), ('rejected', 1, True))
        self.retained(rejected, started['request'])
        snapshot = (await driver.command('diagnostics'))['diagnostics']
        records = [r for r in snapshot['recent_records'] if r['event_name'] == 'rpc_failed']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['subject_id'], self.subject)
        fields = records[0]['fields']
        self.assertEqual({key: fields.get(key) for key in ('operation', 'result', 'grpc_status', 'reason',
                                                          'struct_commands', 'set_fields', 'time_samples', 'limit')},
                         dict(operation=2, result=1, grpc_status=3, reason=1, struct_commands=3, set_fields=2,
                              time_samples=0, limit=4))
        self.assertEqual(self.state.rejections[-1]['kind'], 'delta_too_large')
        self.assertFalse(self.commits())
        await self.quit(driver)

    async def test_creation_prepared_close_retains_seed_without_issuing(self):
        driver, _ = await self.prepare(host=True)
        attempts = dict(self.state.attempts)
        closed = await driver.command('close 5000')
        self.assertEqual((closed['result'], closed['phase'], closed['create_result']), (0, 'unissued', 10), closed)
        self.assertTrue(closed['closing'])
        self.assertTrue(closed['quiescent'])
        self.assertFalse(closed['started'])
        self.assertEqual((closed['object'], closed['sequence'], closed['request']), (0, 0, 0))
        self.retained(closed)
        for command, result in (('start', 10), ('attach', 10), ('duplicate', 5), ('wait 1', 0), ('close 5000', 0)):
            report = await driver.command(command)
            self.assertEqual(report['result'], result, report)
            self.assertEqual((report['phase'], report['create_result'], report['request']), ('unissued', 10, 0))
            self.assertFalse(report['started'])
            self.retained(report)
        self.assertEqual(self.state.attempts, attempts)
        self.assertFalse(self.commits())
        self.assertEqual((await driver.command('workflow-close'))['result'], 0)
        await self.quit(driver)
        self.assertEqual(self.state.attempts, attempts)
        self.assertFalse(self.commits())

    async def prepared_host_replacement(self, command):
        driver, _ = await self.prepare(host=True)
        attempts = dict(self.state.attempts)
        changed = await driver.command(command)
        if command.endswith('deferred-start'):
            self.assertEqual(changed['result'], 5)
        self.assertEqual((changed['phase'], changed['create_result']), ('unissued', 12))
        self.assertTrue(changed['started'])
        self.assertFalse(changed['closing'])
        self.assertEqual((changed['request'], changed['object'], changed['sequence']), (0, 0, 0))
        self.assertEqual((changed['status_state'], changed['status_result'], changed['status_pending'],
                          changed['status_bytes'], changed['status_unconfirmed']), (7, 12, 0, 0, 0))
        self.assertIsNone(changed['diagnostics'])
        self.assertEqual((await driver.command('wait 5000'))['result'], 0)
        await self.observed_status(driver, 7, 12)
        # The caller exports the private host layer with SdfLayer::Export
        # before close, without issuing Create.
        path = self.artifact / (self.id().split('.')[-1] + '.usdc')
        exported = await driver.command('host-export ' + json.dumps(str(path)))
        self.assertEqual(exported['result'], 0, exported)
        self.assertEqual(self.state.attempts, attempts, 'host export issued service requests')
        host = {node['path']: node for node in exported['host']}
        self.assertEqual(host['/']['fields']['hostWitness'], dict(type='int32', value=777))
        self.assertEqual(host['/']['fields']['primChildren'], dict(type='tokens', value=['HostOnly']))
        self.assertEqual(host['/HostOnly']['type'], 6)
        self.assertEqual(host['/HostOnly']['fields']['specifier'], dict(type='specifier', value=0))
        for _ in range(2):
            self.assertEqual((await driver.command('start'))['result'], 5)
        await self.quit(driver)
        self.assertEqual(self.state.attempts, attempts)
        self.assertFalse(self.commits())

    async def test_prepared_host_replacement_consumes_unissued_creation_and_preserves_namespace(self):
        await self.prepared_host_replacement('host-replace')

    async def test_prepared_host_start_observes_replacement_before_deferred_notice(self):
        await self.prepared_host_replacement('host-replace-deferred-start')

    async def test_creation_start_reentrant_host_predicate_cannot_consume_attempt(self):
        driver, _ = await self.prepare(host=True)
        self.assertEqual((await driver.command('start-reentrant'))['result'], 0)
        settled = await driver.command('wait 5000')
        self.assertEqual((settled['result'], settled['phase']), (0, 'confirmed'))
        self.assertEqual((await driver.command('start'))['result'], 5)
        await self.quit(driver)
        self.committed_once(settled['object'])

    async def test_initializing_target_destroyed_during_lost_reply_retains_uncertain_seed(self):
        driver, _ = await self.prepare(target=True)
        fault = self.state.plan('CreateLayer', 'after_commit', pause=True, code=grpc.StatusCode.UNAVAILABLE)
        started = await driver.command('start')
        self.assertEqual(started['result'], 0)
        try:
            await asyncio.wait_for(fault.reached.wait(), 5)
            self.committed_once()
            destroyed = await driver.command('target-destroy')
            self.assertFalse(destroyed['target_alive'])
        finally:
            fault.release.set()
        settled = await driver.command('wait 5000')
        self.assertEqual((settled['result'], settled['phase'], settled['create_result']), (0, 'unknown', 19))
        self.retained(settled, started['request'])
        await self.observed_status(driver, 7, 19)
        await driver.command('replace-host')
        self.assertEqual((await driver.command('start'))['result'], 5, 'uncertain create must never be resubmitted')
        await self.quit(driver)
        self.committed_once()
        self.assertEqual(self.state.reads, [])

    async def test_initializing_target_destruction_keeps_the_confirmed_actor_until_close(self):
        driver, prepared = await self.prepare(target=True)
        receipt = await self.confirmed(driver)
        attached = await driver.command('target-attach')
        self.assertEqual((attached['result'], attached['applied'], attached['journal_applied']), (0, 1, 1))
        self.assertEqual(attached['authoritative'], self.expected)
        self.assertEqual(attached['private_host'], prepared['private_host'])
        await self.observed_status(driver, 1, 0)
        destroyed = await driver.command('target-destroy')
        self.assertFalse(destroyed['target_alive'])
        self.assertEqual((destroyed['status_state'], destroyed['applied']), (1, 1),
                         'target destruction must detach without closing the retained actor')
        failed = await driver.command('target-fail-actor')
        self.assertEqual((failed['result'], failed['status_state'], failed['status_result']), (0, 7, 13))
        self.assertEqual((failed['phase'], failed['object']), ('confirmed', receipt['object']))
        await self.observed_status(driver, 7, 13)
        await driver.command('replace-host')
        # The close reply fences retirement: the retained actor closes only here,
        # keeping its failure cause, and the observer reaches the same revision.
        closed = await driver.command('workflow-close')
        self.assertEqual((closed['result'], closed['phase'], closed['object']), (0, 'confirmed', receipt['object']))
        self.assertTrue(closed['closing'] and closed['quiescent'], closed)
        self.assertEqual((closed['status_state'], closed['status_result']), (9, 13), 'close must retire the actor')
        await self.observed_status(driver, 9, 13)
        await self.quit(driver)
        self.committed_once(receipt['object'])
        self.assertEqual(self.state.reads, [dict(backend='A', attempt=1, object=receipt['object'], cursor=0)])

    async def test_creation_wait_timeout_does_not_cancel_or_replace_receipt(self):
        driver, _ = await self.prepare()
        fault = self.state.plan('CreateLayer', 'after_commit', pause=True)
        started = await driver.command('start')
        self.assertEqual(started['result'], 0)
        try:
            await asyncio.wait_for(fault.reached.wait(), 5)
            waiting = await driver.command('wait 10')
            self.assertEqual(waiting['result'], 9, waiting)
            self.assertEqual(waiting['phase'], 'creating')
            self.assertEqual((waiting['status_state'], waiting['status_pending'], waiting['status_bytes']),
                             (1, 1, len(self.seed)))
            self.assertFalse(waiting['closing'])
            self.assertFalse(waiting['quiescent'])
            self.retained(waiting, started['request'])
            self.assertEqual((await driver.command('start'))['result'], 5)
            self.assertEqual((await driver.command('attach'))['result'], 14)
            self.committed_once()
        finally:
            fault.release.set()
        ready = await driver.command('wait 5000')
        self.assertEqual((ready['result'], ready['phase'], ready['create_result']), (0, 'confirmed', 0))
        self.assertTrue(ready['quiescent'])
        self.retained(ready, started['request'])
        self.committed_once(ready['object'])
        await self.quit(driver)

    async def test_creation_close_issued_before_commit_retains_uncertainty(self):
        driver, _ = await self.prepare()
        # Even if server cancellation arrives late, releasing this gate cannot
        # commit. Client quiescence alone does not promise remote retraction.
        fault = self.state.plan('CreateLayer', 'before', pause=True, code=grpc.StatusCode.UNAVAILABLE)
        started = await driver.command('start')
        self.assertEqual(started['result'], 0)
        try:
            await asyncio.wait_for(fault.reached.wait(), 5)
            closed = await driver.command('close 5000')
            self.assertEqual(closed['result'], 0, closed)
            self.assertEqual((closed['phase'], closed['create_result']), ('unknown', 19))
            self.assertTrue(closed['closing'])
            self.assertTrue(closed['quiescent'])
            self.retained(closed, started['request'])
            self.assertEqual((await driver.command('start'))['result'], 5)
            self.assertEqual((await driver.command('attach'))['result'], 10)
            self.assertEqual((await driver.command('duplicate'))['result'], 5)
            self.assertFalse(self.commits())
            self.assertEqual(self.state.attempts.get('CreateLayer'), 1)
        finally:
            fault.release.set()
        await self.quit(driver)
        self.assertFalse(self.commits())
        self.assertEqual(self.state.attempts.get('CreateLayer'), 1)

    async def test_creation_replacement_before_attach_keeps_confirmed_object(self):
        driver, _ = await self.prepare()
        ready = await self.confirmed(driver)
        replacement = encode(Delta(commands=(C(1, 77, 1, 'Replacement', 6),), fields=(F(77, 'counter', 999),)))
        frames = [pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name=self.layer)),
                  pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=replacement))]
        new_object = (await self.stub.CreateLayer(iter(frames), metadata=AUTH, timeout=5)).object_id
        self.assertNotEqual(new_object, ready['object'])
        failed = await driver.command('attach')
        self.assertEqual((failed['result'], failed['attach_result']), (16, 16), failed)
        self.assertEqual((failed['phase'], failed['create_result'], failed['object']), ('confirmed', 0, ready['object']))
        self.assertEqual(failed['applied'], 0)
        self.retained(failed, ready['request'])
        self.assertEqual(self.state.reads, [dict(backend='A', attempt=1, object=ready['object'], cursor=0)])
        self.assertEqual(self.state.attempts.get('CreateLayer'), 2)
        self.assertEqual([entry.payload for entry in self.commits()], [self.seed, replacement])
        await self.quit(driver)
        self.assertEqual(self.state.attempts.get('CreateLayer'), 2)

    async def test_creation_ordinary_sequence_one_bootstrap(self):
        self.ordinary_seed()
        address, _ = await self.backend(ReferenceServer)
        driver, _ = await self.prepare(address)
        ready = await self.confirmed(driver)
        attached = await driver.command('attach')
        self.assertEqual((attached['result'], attached['attach_result']), (0, 0), attached)
        self.assertTrue(attached['attached'])
        self.assertEqual((attached['attach_object'], attached['applied']), (ready['object'], 1))
        self.assertEqual(attached['authoritative'], self.expected)
        self.assertEqual(attached['visible'], self.expected)
        self.retained(attached, ready['request'])
        self.committed_once(ready['object'])
        self.assertEqual(self.state.attempts.get('SubscribeReadLayer'), 1)
        await self.quit(driver)

    def host_expected(self):
        projected = []
        for node in self.expected:
            record = {key: copy.deepcopy(node[key]) for key in ('path', 'type', 'fields')}
            # Public SdfLayer field enumeration includes schema-required
            # fallbacks absent from the exact retained backing data.
            if record['type'] == 6:
                record['fields'].setdefault('specifier', {'type': 'specifier', 'value': 1})
            projected.append(record)
        return sorted(projected, key=lambda node: node['path'])

    async def host_until(self, driver, predicate):
        async def poll():
            while True:
                report = await driver.command('pump')
                self.assertEqual(report['result'], 0, report)
                if predicate(report):
                    return report
                await asyncio.sleep(0.005)
        return await asyncio.wait_for(poll(), 8)

    async def host_staged(self, driver, receipt):
        staged = await driver.command('host-prepare 5000')
        self.assertEqual((staged['result'], staged['attach_result']), (0, 0), staged)
        self.assertEqual((staged['phase'], staged['create_result'], staged['object']),
                         ('confirmed', 0, receipt['object']))
        self.assertEqual((staged['applied'], staged['journal_applied'], staged['status_state']), (0, 1, 1))
        self.assertFalse(staged['layer_exposed'])
        self.assertFalse(staged['attached'])
        self.assertFalse(staged['registered'])
        self.assertEqual(staged['identifier_notices'], 0)
        self.assertEqual(staged['host'], [])
        self.assertEqual(staged['authoritative'], self.expected)
        self.assertEqual(staged['visible'], self.expected)
        self.retained(staged, receipt['request'])
        return staged

    async def host_activated(self, driver, receipt):
        activated = await driver.command('activate')
        self.assertEqual((activated['result'], activated['attach_result']), (0, 0), activated)
        self.assertTrue(activated['layer_exposed'])
        self.assertTrue(activated['attached'])
        self.assertTrue(activated['registered'])
        self.assertEqual(activated['identifier_notices'], 1)
        self.assertEqual(activated['host'], self.host_expected())
        ready = await self.host_until(driver, lambda report: report['applied'] == 1 and report['status_state'] == 2)
        self.assertEqual((ready['attach_object'], ready['journal_applied']), (receipt['object'], 1))
        self.assertTrue(ready['registered'])
        self.assertEqual(ready['identifier_notices'], 1)
        self.assertEqual(ready['host'], self.host_expected())
        self.assertEqual(ready['authoritative'], self.expected)
        self.assertEqual(ready['visible'], self.expected)
        self.retained(ready, receipt['request'])
        await self.observed_status(driver, 2, 0)
        rpc_counts = {name: self.state.attempts.get(name, 0) for name in ('CreateLayer', 'SubscribeReadLayer')}
        repeated = await driver.command('activate')
        self.assertEqual((repeated['result'], repeated['attach_result']), (0, 0), repeated)
        self.assertTrue(repeated['registered'])
        self.assertTrue(repeated['layer_exposed'])
        self.assertEqual(repeated['identifier_notices'], 1)
        self.assertEqual(repeated['host'], self.host_expected())
        self.retained(repeated, receipt['request'])
        self.assertEqual({name: self.state.attempts.get(name, 0) for name in rpc_counts}, rpc_counts)
        return ready

    async def test_host_creation_freezes_context_before_rpc_and_activates_graph_bootstrap(self):
        driver, _ = await self.prepare(host=True)
        # The global replacement rejects use. Both bootstrap and activation
        # must retain the context selected during preparation, before Create.
        self.assertEqual((await driver.command('replace-host'))['result'], 0)
        receipt = await self.confirmed(driver)
        await self.host_staged(driver, receipt)
        await self.host_staged(driver, receipt)
        self.assertEqual(self.state.attempts.get('SubscribeReadLayer'), 1)
        await self.host_activated(driver, receipt)
        self.assertEqual(self.state.reads, [dict(backend='A', attempt=1, object=receipt['object'], cursor=0)])
        closed_creation = await driver.command('close 5000')
        self.assertEqual(closed_creation['result'], 0, closed_creation)
        self.assertEqual((closed_creation['status_state'], closed_creation['status_result']), (2, 0))
        self.assertTrue(closed_creation['registered'])
        self.assertEqual(closed_creation['host'], self.host_expected())
        self.retained(closed_creation, receipt['request'])
        self.committed_once(receipt['object'])
        await self.quit(driver)

    async def test_host_creation_ordinary_sequence_one_bootstrap(self):
        self.ordinary_seed()
        address, _ = await self.backend(ReferenceServer)
        driver, _ = await self.prepare(address, host=True)
        receipt = await self.confirmed(driver)
        await self.host_staged(driver, receipt)
        await self.host_activated(driver, receipt)
        self.assertEqual(self.state.attempts.get('SubscribeReadLayer'), 1)
        self.committed_once(receipt['object'])
        await self.quit(driver)

    async def test_hook_failure_after_activation_leaves_the_layer_ready(self):
        driver, _ = await self.prepare(host=True)
        receipt = await self.confirmed(driver)
        await self.host_staged(driver, receipt)
        await self.host_activated(driver, receipt)
        attempts = dict(self.state.attempts)
        # Activation confirmed and published the creation: a hook failure
        # after it (a late Export deadline) is refused and changes nothing.
        refused = await driver.command('hook-fail-timeout')
        self.assertEqual((refused['result'], refused['status_state'], refused['status_result']), (1, 2, 0), refused)
        await self.observed_status(driver, 2, 0)
        self.assertEqual(self.state.attempts, attempts)
        self.committed_once(receipt['object'])
        await self.quit(driver)

    async def test_host_creation_prepare_timeout_retains_confirmed_seed_and_subscription(self):
        driver, _ = await self.prepare(host=True)
        receipt = await self.confirmed(driver)
        fault = self.state.plan('SubscribeReadLayer', 'before', pause=True)
        try:
            waiting = await driver.command('host-prepare 10')
            self.assertEqual(waiting['result'], 9, waiting)
            await asyncio.wait_for(fault.reached.wait(), 5)
            self.assertEqual((waiting['phase'], waiting['create_result'], waiting['object']),
                             ('confirmed', 0, receipt['object']))
            self.assertEqual(waiting['applied'], 0)
            self.assertEqual(waiting['status_state'], 1)
            self.assertFalse(waiting['layer_exposed'])
            self.assertFalse(waiting['registered'])
            self.assertEqual(waiting['identifier_notices'], 0)
            self.assertEqual(waiting['host'], [])
            self.retained(waiting, receipt['request'])
            self.assertEqual(self.state.attempts.get('SubscribeReadLayer'), 1)
            self.committed_once(receipt['object'])
        finally:
            fault.release.set()
        await self.host_staged(driver, receipt)
        await self.host_activated(driver, receipt)
        self.assertEqual(self.state.attempts.get('SubscribeReadLayer'), 1)
        self.committed_once(receipt['object'])
        await self.quit(driver)

    async def test_host_creation_replacement_before_bootstrap_preserves_receipt(self):
        driver, _ = await self.prepare(host=True)
        receipt = await self.confirmed(driver)
        replacement = encode(Delta(commands=(C(1, 77, 1, 'Replacement', 6),), fields=(F(77, 'counter', 999),)))
        frames = [pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name=self.layer)),
                  pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=replacement))]
        new_object = (await self.stub.CreateLayer(iter(frames), metadata=AUTH, timeout=5)).object_id
        self.assertNotEqual(new_object, receipt['object'])
        failed = await driver.command('host-prepare 5000')
        self.assertEqual((failed['result'], failed['attach_result']), (16, 16), failed)
        self.assertEqual((failed['phase'], failed['create_result'], failed['object']),
                         ('confirmed', 0, receipt['object']))
        self.assertFalse(failed['layer_exposed'])
        self.assertFalse(failed['attached'])
        self.assertFalse(failed['registered'])
        self.assertEqual(failed['identifier_notices'], 0)
        self.assertEqual(failed['host'], [])
        self.assertEqual(failed['applied'], 0)
        self.retained(failed, receipt['request'])
        self.assertEqual(self.state.reads, [dict(backend='A', attempt=1, object=receipt['object'], cursor=0)])
        self.assertEqual([entry.payload for entry in self.commits()], [self.seed, replacement])
        self.assertEqual(self.state.attempts.get('CreateLayer'), 2)
        current = await driver.command('report')
        self.assertEqual(current['applied'], 0)
        self.assertFalse(current['layer_exposed'])
        await self.terminal_status(driver, current['status_result'])
        await self.quit(driver)
        self.assertEqual(self.state.attempts.get('CreateLayer'), 2)
        self.assertEqual([entry.payload for entry in self.commits()], [self.seed, replacement])

    async def test_host_creation_peer_dispatch_failure_retains_original_seed(self):
        driver, _ = await self.prepare(host=True)
        receipt = await self.confirmed(driver)
        await self.host_staged(driver, receipt)
        await self.host_activated(driver, receipt)
        self.assertEqual((await driver.command('host-reject-dispatch'))['result'], 0)
        payload = encode(Delta(fields=(F(3, 'weight', 91.5),)))
        frames = [pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name=self.layer,
                    object_id=receipt['object'], client_id='creation-peer')),
                  pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=payload))]
        sequence = (await self.stub.UpdateLayer(iter(frames), metadata=AUTH, timeout=5)).sequence_num
        self.assertEqual(sequence, 2)
        failed = await self.host_until(driver, lambda report:
                                      report['attach_result'] == 13 and report['status_state'] == 7)
        self.assertEqual(failed['status_state'], 7)
        self.assertEqual(failed['applied'], 1)
        self.assertTrue(failed['layer_exposed'])
        self.assertTrue(failed['registered'])
        self.assertEqual(failed['identifier_notices'], 1)
        self.assertEqual(failed['host'], self.host_expected())
        self.assertEqual((failed['phase'], failed['create_result'], failed['object']),
                         ('confirmed', 0, receipt['object']))
        self.retained(failed, receipt['request'])
        self.assertEqual([entry.payload for entry in self.commits()], [self.seed, payload])
        self.assertEqual(self.state.attempts.get('CreateLayer'), 1)
        self.assertEqual(self.state.attempts.get('UpdateLayer'), 1)
        self.assertEqual(self.state.attempts.get('SubscribeReadLayer'), 1)
        await self.terminal_status(driver, 13)
        await self.quit(driver)
        self.assertEqual([entry.payload for entry in self.commits()], [self.seed, payload])

    async def test_host_creation_activation_collision_preserves_external_layer_and_seed(self):
        driver, _ = await self.prepare(host=True)
        receipt = await self.confirmed(driver)
        await self.host_staged(driver, receipt)
        occupied = await driver.command('occupy')
        self.assertEqual(occupied['result'], 0, occupied)
        self.assertTrue(occupied['registered'])
        self.assertFalse(occupied['layer_exposed'])
        external = occupied['registered_host']
        self.assertNotEqual(external, self.host_expected())
        self.assertFalse(any(node['path'] == '/P' for node in external))
        counts = {name: self.state.attempts.get(name, 0) for name in ('CreateLayer', 'SubscribeReadLayer')}
        for _ in range(2):
            failed = await driver.command('activate')
            self.assertEqual(failed['result'], 5, failed)
            self.assertFalse(failed['layer_exposed'])
            self.assertFalse(failed['attached'])
            self.assertTrue(failed['registered'])
            self.assertEqual(failed['registered_host'], external)
            self.assertEqual((failed['phase'], failed['create_result'], failed['object']),
                             ('confirmed', 0, receipt['object']))
            self.retained(failed, receipt['request'])
            self.assertEqual({name: self.state.attempts.get(name, 0) for name in counts}, counts)
        await self.terminal_status(driver, 5)
        self.committed_once(receipt['object'])
        await self.quit(driver)

    async def test_host_creation_activation_metadata_failure_preserves_published_content_if_registered(self):
        driver, _ = await self.prepare(host=True)
        receipt = await self.confirmed(driver)
        await self.host_staged(driver, receipt)
        counts = {name: self.state.attempts.get(name, 0) for name in ('CreateLayer', 'SubscribeReadLayer')}
        stat_attempts = self.state.attempts.get('StatLayer', 0)
        # The creation's status subject is the existence evidence for every
        # resolution of its URI, so the pinned SDK's one metadata read is the
        # rename's modification time. Fail it to exercise failure after
        # registration.
        fault = self.state.plan('StatLayer', 'before', attempt=stat_attempts + 1,
                                code=grpc.StatusCode.PERMISSION_DENIED)
        first = await driver.command('activate')
        await asyncio.wait_for(fault.reached.wait(), 5)
        self.assertNotEqual(first['result'], 0, first)
        first_stat_attempts = self.state.attempts.get('StatLayer', 0)
        for failed in (first, await driver.command('activate')):
            self.assertEqual(failed['result'], first['result'])
            self.assertFalse(failed['layer_exposed'])
            self.assertFalse(failed['attached'])
            self.assertEqual((failed['phase'], failed['create_result'], failed['object']),
                             ('confirmed', 0, receipt['object']))
            self.retained(failed, receipt['request'])
            self.assertTrue(failed['registered'])
            self.assertEqual(failed['registered_host'], self.host_expected(),
                             'failed identifier publication exposed empty or partial canonical contents')
            self.assertEqual({name: self.state.attempts.get(name, 0) for name in counts}, counts)
        self.assertEqual(self.state.attempts.get('StatLayer'), first_stat_attempts)
        await self.terminal_status(driver, first['result'])
        self.committed_once(receipt['object'])
        await self.quit(driver)

    async def test_host_creation_external_failure_cancels_queued_peer_publication(self):
        driver, _ = await self.prepare(host=True)
        receipt = await self.confirmed(driver)
        await self.host_staged(driver, receipt)
        await self.host_activated(driver, receipt)
        payload = encode(Delta(fields=(F(3, 'weight', 123.5),)))
        frames = [pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name=self.layer,
                    object_id=receipt['object'], client_id='queued-creation-peer')),
                  pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=payload))]
        sequence = (await self.stub.UpdateLayer(iter(frames), metadata=AUTH, timeout=5)).sequence_num
        self.assertEqual(sequence, 2)

        async def held_host_job():
            while True:
                report = await driver.command('report')  # Never pump before injecting failure.
                if report['host_queued'] == 1 and report['covered'] == 2:
                    return report
                await asyncio.sleep(0.005)
        held = await asyncio.wait_for(held_host_job(), 8)
        self.assertEqual((held['applied'], held['journal_applied'], held['retained']), (1, 1, 1))
        self.assertEqual(held['host'], self.host_expected())
        self.assertEqual(held['authoritative'], self.expected)
        self.assertEqual(held['visible'], self.expected)
        failed = await driver.command('fail-publication')
        self.assertEqual(failed['result'], 0, failed)
        self.assertEqual(failed['host_queued'], 1)
        self.retained(failed, receipt['request'])
        # This explicitly executes the queued closure after terminal failure.
        # It must retire its owner without installing its peer candidate.
        late = await driver.command('pump')
        self.assertEqual(late['result'], 0, late)
        self.assertEqual(late['host_queued'], 0)
        failed = await self.terminal_status(driver, 13)
        for state in (late, failed):
            self.assertEqual((state['applied'], state['journal_applied'], state['retained']), (1, 1, 1))
            self.assertEqual(state['host'], self.host_expected())
            self.assertEqual(state['authoritative'], self.expected)
            self.assertEqual(state['visible'], self.expected)
            self.assertEqual(state['identifier_notices'], 1)
            self.retained(state, receipt['request'])
        self.assertEqual(len(failed['retained_peers']), 1)
        peer = failed['retained_peers'][0]
        self.assertGreater(peer['id'], 0)
        self.assertEqual((peer['first'], peer['last'], peer['applied']), (2, 2, False))
        self.assertEqual(bytes.fromhex(peer['bytes']), payload)
        self.assertEqual([entry.payload for entry in self.commits()], [self.seed, payload])
        self.assertEqual({name: self.state.attempts.get(name) for name in
                          ('CreateLayer', 'SubscribeReadLayer', 'UpdateLayer')},
                         dict(CreateLayer=1, SubscribeReadLayer=1, UpdateLayer=1))
        await self.quit(driver)
