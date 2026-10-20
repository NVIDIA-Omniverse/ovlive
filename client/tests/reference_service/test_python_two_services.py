# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two independent live services, one endpoint each, in one Python USD host runtime and one stage."""
import asyncio
import json
import os
from pathlib import Path
import sys
import unittest

import grpc
from omni.usd import live
from service import Principal, pb, rpc
from graph_service import GraphState, GraphServer
import scene_graph_model as graph_model
from scene_graph_wire import decode as graph_decode
from scene_wire import empty_snapshot, update
import test_scene as scene_support

SERVICES = ('A', 'B')
LAYER = 'root.live'
READY, AUTH_WAIT = int(live.State.READY), int(live.State.AUTH_WAIT)
REJECTED = 3  # Public OMNI_USD_LIVE_AUTH_REJECTED.
LAYER_SUBJECT = 3  # Public OMNI_USD_LIVE_SUBJECT_LAYER.


class Service:
    """One graph reference service with its own principal, token, session and layer.

    The graph service carries root fields and prims, so one stage composes
    prim opinions from both services' layers.
    """
    def __init__(self, name, seed):
        self.name, self.token = name, f'token-{name.lower()}'
        self.state = GraphState(seed)
        self.state.tokens = {self.token: Principal(f'writer-{name.lower()}')}
        self.server = GraphServer(self.state, name)

    @property
    def metadata(self):
        return (('authorization', f'Bearer {self.token}'),)

    async def start(self):
        self.address = await self.server.start()
        self.channel = grpc.aio.insecure_channel(self.address)
        await asyncio.wait_for(self.channel.channel_ready(), 5)
        self.stub = rpc.LiveServiceStub(self.channel)
        self.session = (await self.stub.CreateSession(pb.CreateSessionRequest(path='two-services', name='shared'),
                                                     metadata=self.metadata, timeout=5)).session_id
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name=LAYER)),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=empty_snapshot()))]),
            metadata=self.metadata, timeout=5)
        self.uri = live.make_layer_uri(live.make_session_uri(self.address, self.session), LAYER)

    async def peer(self, fields):
        await self.stub.UpdateLayer(iter([
            pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name=LAYER,
                                                             object_id=1, client_id='independent-peer')),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=update(fields)))]),
            metadata=self.metadata, timeout=5)

    def commits(self):
        return self.state.sessions[self.session].layers[LAYER].commits

    def model(self):
        return self.state.sessions[self.session].layers[LAYER].model

    def oracle(self):
        """The service's root metadata fields; prims are in prims()."""
        fields = next(node for node in self.model().state if node['id'] == 1)['fields']
        return {key: value for key, value in fields.items() if key != 'primChildren'}

    def prims(self):
        """The default of each prim's `value` attribute in the service, by prim path."""
        nodes = self.model().state
        paths = {node['id']: node['path'] for node in nodes}
        return {paths[node['parent']]: node['fields'].get('default') for node in nodes
                if node['type'] == graph_model.ATTRIBUTE and node['name'] == 'value'}

    def updates(self):
        """(writer, root fields) of every committed update after the layer's creation.

        An update with commands, samples or non-root fields reads as None.
        """
        result = []
        for commit in self.commits()[1:]:
            delta = graph_decode(commit.payload)
            structural = delta.commands or delta.samples or any(field.section != 1 for field in delta.fields)
            result.append((commit.writer, None if structural else {f.key: f.value for f in delta.fields}))
        return result


class PythonTwoServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.seed = int(os.environ['LIVE_TEST_SEED'])
        self.artifact = Path(os.environ['LIVE_TEST_ARTIFACT_DIR'])
        self.driver = None
        self.services = {}
        self.addAsyncCleanup(self.cleanup)
        # Equal seeds give both services the same session ID namespace, so
        # their sessions and layers share an ID and a name.
        self.services = {name: Service(name, self.seed) for name in SERVICES}
        for service in self.services.values():
            await service.start()
        self.a, self.b = self.services['A'], self.services['B']

    async def cleanup(self):
        failures = []
        if self.driver is not None and self.driver.process.returncode is None:
            try:
                self.driver.process.kill()
            except ProcessLookupError:
                pass
        operations = [*((self.driver.process.wait(), self.driver.drainer) if self.driver else ()),
                      *(service.channel.close() for service in self.services.values() if hasattr(service, 'channel')),
                      *(service.server.stop() for service in self.services.values())]
        results = await asyncio.wait_for(asyncio.gather(*operations, return_exceptions=True), 10)
        failures.extend(result for result in results if isinstance(result, BaseException))
        for service in self.services.values():
            try:
                service.state.assert_faults_fired()
            except AssertionError as error:
                failures.append(error)
        (self.artifact / (self.id().split('.')[-1] + '.two-services.json')).write_text(json.dumps(
            {'seed': self.seed, 'services': {name: dict(ledger=service.state.snapshot(), attempts=service.state.attempts)
                                             for name, service in self.services.items()},
             'cleanup_errors': [repr(error) for error in failures]}, indent=2, sort_keys=True) + '\n')
        self.assertFalse(failures, failures)

    async def start(self):
        extension = Path(os.environ['LIVE_TEST_PYTHON_EXTENSION'])
        environment = dict(os.environ, PYTHONPATH=str(extension.parents[3]) + os.pathsep + os.environ['PYTHONPATH'],
                           PYTHONDONTWRITEBYTECODE='1')
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).parents[1] / 'native/python/two_services.py'), str(extension.parents[4]),
            *(f'{name}={service.address}={service.token}' for name, service in self.services.items()),
            env=environment, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT)
        self.driver = scene_support.Driver(process, self.artifact / f'{self.id().split(".")[-1]}.two-services.log')
        ready = await self.driver.receive()
        self.assertEqual(ready['type'], 'ready', ready)
        self.handles = ready['endpoints']
        self.assertEqual(len(set(self.handles.values())), 2, 'two authorities are two endpoints')
        return self.driver

    async def open_stage(self, initial):
        """Seed each service's layer with a peer edit, then open both in one stage."""
        for name, service in self.services.items():
            await service.peer(initial[name])
        driver = await self.start()
        opened = await driver.command(f'open-stage {self.a.uri} {self.b.uri}')
        self.assertEqual(opened['stack'][1:], [self.a.uri, self.b.uri], opened)
        for name, service in self.services.items():
            layer = opened['layers'][service.uri]
            self.assertEqual((layer['state'], layer['fields']), (READY, initial[name]), layer)
        return driver

    async def author(self, service, key, value):
        """Author one field on the service's live layer and save until the service confirms it."""
        await self.driver.command(f'set {service.uri} {key} {value}')
        self.assertEqual((await self.driver.command(f'save {service.uri} 5000'))['result'], int(live.Result.OK))

    async def define(self, service, prim, value):
        """Define a prim with an int `value` attribute on the service's live layer and save until confirmed."""
        await self.driver.command(f'define {service.uri} {prim} {value}')
        self.assertEqual((await self.driver.command(f'save {service.uri} 5000'))['result'], int(live.Result.OK))
        self.assertEqual(service.prims().get(prim), value)

    async def composed(self, prim):
        """The stage's composed `value` of the prim and the layer of its strongest opinion."""
        result = await self.driver.command(f'composed {prim}')
        return result['value'], result['layer']

    async def read(self, service, expected):
        """A peer edit on the service reaches its live layer and only that layer."""
        result = await self.driver.command(f'wait {service.uri}' + ''.join(f' {k} {v}' for k, v in expected.items()))
        self.assertEqual((result['state'], result['fields']), (READY, expected), result)

    async def close(self):
        self.assertEqual((await self.driver.command('close'))['type'], 'closed')
        await asyncio.wait_for(self.driver.process.wait(), 10)
        await self.driver.drainer
        self.assertIsNone(self.driver.failure)
        self.assertEqual(self.driver.process.returncode, 0, self.driver.log.read_text()[-8192:])

    def host_writers(self):
        """The one wire identity each service saw commit the host's edits."""
        writers = {}
        for name, service in self.services.items():
            identities = {writer for writer, _ in service.updates() if writer != 'independent-peer'}
            self.assertEqual(len(identities), 1, (name, identities))
            writers[name] = identities.pop()
        return writers

    async def test_layers_of_two_services_author_read_and_publish_independently_in_one_stage(self):
        # Same session ID and layer name on both services, under different authorities.
        self.assertEqual(self.a.session, self.b.session)
        parts = {name: live.parse_uri(service.uri) for name, service in self.services.items()}
        self.assertEqual({(p.session_id, p.layer_name) for p in parts.values()}, {(self.a.session, LAYER)})
        self.assertNotEqual(parts['A'].authority, parts['B'].authority)
        self.assertNotIn(self.a.token, self.b.state.tokens)  # Each service accepts only its own token.
        self.assertNotIn(self.b.token, self.a.state.tokens)

        await self.open_stage({'A': {'framesPerSecond': 24.0}, 'B': {'framesPerSecond': 30.0}})
        expected = {'A': {'framesPerSecond': 24.0}, 'B': {'framesPerSecond': 30.0}}
        # Interleaved host edits publish to their own service only.
        for name, key, value in (('A', 'startTimeCode', 1.0), ('B', 'startTimeCode', 2.0),
                                 ('B', 'endTimeCode', 20.0), ('A', 'endTimeCode', 10.0)):
            await self.author(self.services[name], key, value)
            expected[name][key] = value
        # Peer edits on each service reach only that service's layer.
        await self.a.peer({'timeCodesPerSecond': 48.0})
        expected['A']['timeCodesPerSecond'] = 48.0
        await self.read(self.a, expected['A'])
        await self.b.peer({'timeCodesPerSecond': 60.0})
        expected['B']['timeCodesPerSecond'] = 60.0
        await self.read(self.b, expected['B'])
        await self.read(self.a, expected['A'])

        writers = self.host_writers()
        self.assertNotEqual(writers['A'], writers['B'], 'each endpoint has its own writer ID')
        for name, service in self.services.items():
            self.assertEqual(service.oracle(), expected[name])
            self.assertEqual(service.updates(), [
                ('independent-peer', {'framesPerSecond': expected[name]['framesPerSecond']}),
                *(((writers[name], {'startTimeCode': 1.0}), (writers[name], {'endTimeCode': 10.0}))
                  if name == 'A' else ((writers[name], {'startTimeCode': 2.0}), (writers[name], {'endTimeCode': 20.0}))),
                ('independent-peer', {'timeCodesPerSecond': expected[name]['timeCodesPerSecond']})])
            self.assertEqual(service.state.auth_rejections, [])
        states = {name: await self.driver.command(f'state {service.uri}') for name, service in self.services.items()}
        self.assertNotEqual(states['A']['subject'], states['B']['subject'], 'two independent backings')

        # Status attributes each layer to its own endpoint.
        for name, service in self.services.items():
            report = await self.driver.command(f'wait-status-event {name} {service.uri} {READY}')
            self.assertEqual({event['endpoint'] for event in report['statuses']}, {self.handles[name]}, report)
            layers = {event['uri']: event['subject'] for event in report['statuses']
                      if event['uri'] in (self.a.uri, self.b.uri)}
            self.assertEqual(layers, {service.uri: states[name]['subject']}, report)
            self.assertEqual((report['status']['state'], report['challenges']), (READY, []), report)
        # Diagnostics attribute each layer to its own endpoint.
        canonical = {name: parts[name].canonical_uri for name in SERVICES}
        everything = await self.driver.command('diagnostics')
        self.assertEqual({status['canonical_resource']: status['endpoint'] for status in everything['statuses']
                          if status['subject_kind'] == LAYER_SUBJECT},
                         {canonical[name]: self.handles[name] for name in SERVICES}, everything)
        for name in SERVICES:
            other = canonical['B' if name == 'A' else 'A']
            snapshot = await self.driver.command(f'diagnostics {name}')
            self.assertEqual(snapshot['endpoint_filter'], self.handles[name])
            self.assertEqual({status['endpoint'] for status in snapshot['statuses']}, {self.handles[name]}, snapshot)
            self.assertEqual([status['canonical_resource'] for status in snapshot['statuses']
                              if status['subject_kind'] == LAYER_SUBJECT], [canonical[name]], snapshot)
            self.assertTrue(snapshot['records'], snapshot)
            self.assertEqual({record['endpoint'] for record in snapshot['records']}, {self.handles[name]}, snapshot)
            self.assertNotIn(other, {record['resource'] for record in snapshot['records']}, snapshot)
            self.assertIn(canonical[name], {record['resource'] for record in snapshot['records']}, snapshot)

        # The stage composes one prim from both services' layers; the first
        # sublayer, service A's, holds the strongest opinion.
        await self.define(self.b, '/Shared', 2)
        self.assertEqual(await self.composed('/Shared'), (2, self.b.uri))
        await self.define(self.a, '/Shared', 1)
        self.assertEqual(await self.composed('/Shared'), (1, self.a.uri))
        self.assertEqual((self.a.prims(), self.b.prims()), ({'/Shared': 1}, {'/Shared': 2}))
        await self.close()

    async def test_auth_challenge_on_one_service_does_not_stall_or_fail_the_other(self):
        await self.open_stage({'A': {'framesPerSecond': 24.0}, 'B': {'framesPerSecond': 30.0}})
        # Service A stops accepting its token; the next A request is rejected.
        self.a.state.issue('token-a2', 'writer-a')
        self.a.state.expire(self.a.token)
        await self.driver.command(f'set {self.a.uri} startTimeCode 1.0')
        challenged = await self.driver.command('wait-challenge A')
        self.assertEqual([(c['endpoint'], c['reason'], c['rejected_generation'], c['grpc_status'])
                          for c in challenged['challenges']], [(self.handles['A'], REJECTED, 1, 16)], challenged)
        self.assertEqual((await self.driver.command(f'wait-state {self.a.uri} {AUTH_WAIT}'))['state'], AUTH_WAIT)
        self.assertEqual([(r['method'], r['reason']) for r in self.a.state.auth_rejections],
                         [('UpdateLayer', 'expired')])
        self.assertEqual(self.a.updates(), [('independent-peer', {'framesPerSecond': 24.0})])

        # While A's challenge is unanswered, B reads, authors and publishes.
        await self.b.peer({'endTimeCode': 20.0})
        await self.read(self.b, {'framesPerSecond': 30.0, 'endTimeCode': 20.0})
        await self.author(self.b, 'startTimeCode', 2.0)
        await self.read(self.b, {'framesPerSecond': 30.0, 'endTimeCode': 20.0, 'startTimeCode': 2.0})
        self.assertEqual(self.b.oracle(), {'framesPerSecond': 30.0, 'endTimeCode': 20.0, 'startTimeCode': 2.0})
        other = await self.driver.command('endpoint B')
        self.assertEqual((other['auth']['generation'], other['auth']['challenge_id'], other['challenges']),
                         (1, 0, []), other)
        self.assertEqual(other['status']['state'], READY, other)
        waiting = await self.driver.command('endpoint A')
        self.assertEqual((waiting['status']['state'], waiting['auth']['generation']), (AUTH_WAIT, 1), waiting)
        self.assertEqual((await self.driver.command(f'state {self.a.uri}'))['state'], AUTH_WAIT)

        # The refreshed token reissues A's parked edit once.
        refreshed = await self.driver.command('refresh A token-a2')
        self.assertEqual((refreshed['result'], refreshed['generation']), (int(live.Result.OK), 2), refreshed)
        self.assertEqual((await self.driver.command(f'save {self.a.uri} 5000'))['result'], int(live.Result.OK))
        await self.read(self.a, {'framesPerSecond': 24.0, 'startTimeCode': 1.0})
        writers = self.host_writers()
        self.assertEqual(self.a.updates(), [('independent-peer', {'framesPerSecond': 24.0}),
                                            (writers['A'], {'startTimeCode': 1.0})])
        self.assertEqual(self.a.state.attempts['UpdateLayer'], 3)  # Peer, rejected issue and its single reissue.
        self.assertEqual(self.b.state.auth_rejections, [])
        self.assertEqual(self.b.updates(), [('independent-peer', {'framesPerSecond': 30.0}),
                                            ('independent-peer', {'endTimeCode': 20.0}),
                                            (writers['B'], {'startTimeCode': 2.0})])
        for name in SERVICES:
            report = await self.driver.command(f'endpoint {name}')
            self.assertEqual(report['status']['state'], READY, report)
        self.assertEqual(len((await self.driver.command('endpoint A'))['challenges']), 1)
        self.assertEqual((await self.driver.command('endpoint B'))['challenges'], [])
        await self.close()

    async def test_unregistering_one_endpoint_leaves_the_other_layers_working(self):
        await self.open_stage({'A': {'framesPerSecond': 24.0}, 'B': {'framesPerSecond': 30.0}})
        await self.author(self.a, 'startTimeCode', 1.0)
        await self.define(self.a, '/Shared', 1)
        await self.define(self.b, '/Shared', 2)
        self.assertEqual(await self.composed('/Shared'), (1, self.a.uri))
        released = await self.driver.command(f'release {self.a.uri}')
        self.assertEqual(released['stack'][1:], [self.b.uri], released)
        # With A's sublayer released, the stage composes the prim from B's layer.
        self.assertEqual(await self.composed('/Shared'), (2, self.b.uri))
        self.assertEqual((await self.driver.command('unregister A'))['type'], 'unregistered')
        # An unregistered authority does not resolve, and its service records no further commit.
        self.assertEqual((await self.driver.command(f'stat {self.a.uri}'))['result'], int(live.Result.NOT_FOUND))
        a_commits = len(self.a.state.ledger)

        await self.b.peer({'endTimeCode': 20.0})
        await self.read(self.b, {'framesPerSecond': 30.0, 'endTimeCode': 20.0})
        await self.author(self.b, 'startTimeCode', 2.0)
        await self.read(self.b, {'framesPerSecond': 30.0, 'endTimeCode': 20.0, 'startTimeCode': 2.0})
        self.assertEqual(self.b.oracle(), {'framesPerSecond': 30.0, 'endTimeCode': 20.0, 'startTimeCode': 2.0})
        self.assertEqual((await self.driver.command('endpoint B'))['status']['state'], READY)
        self.assertEqual((await self.driver.command(f'stat {self.b.uri}'))['result'], int(live.Result.OK))
        self.assertEqual(len(self.a.state.ledger), a_commits)
        self.assertEqual(self.a.oracle(), {'framesPerSecond': 24.0, 'startTimeCode': 1.0})
        await self.close()
