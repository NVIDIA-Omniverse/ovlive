# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh Python/pxr host that uses two live services, one endpoint each, in one runtime and one stage.

Arguments: the package root, then one `name=authority=token` per endpoint.
Commands arrive one per line on stdin and each answers one JSON line.
"""
import concurrent.futures
from dataclasses import asdict
import gc
import importlib.util
import json
import queue
import sys
import threading
import time
from pathlib import Path

from pxr import Plug, Sdf, Usd
from omni.usd import live

# Public OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES export flag.
RESOURCE_NAMES = 1


def fields(layer):
    return {key: layer.pseudoRoot.GetInfo(key) for key in layer.pseudoRoot.ListInfoKeys()}


def until(predicate):
    # Failure-only watchdog, short enough that the host reports a hang before
    # the test's command timeout does.
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "two-service host progress deadline"
        time.sleep(.001)


def layer_state(uri):
    """The layer's public state, or None while the runtime tracks no layer of the URI."""
    try:
        return live.get_layer_status(uri).state
    except live.LiveError:
        return None


class Host:
    """One host thread that owns every USD call and runs the dispatched publication jobs."""
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while True:
            action = self.queue.get()
            if action is None:
                return
            action()

    def call(self, action):
        future = concurrent.futures.Future()
        def run():
            try:
                future.set_result(action())
            except BaseException as error:
                future.set_exception(error)
        self.queue.put(run)
        return future.result(8)

    def dispatch(self, action):
        self.queue.put(action)


class Observer:
    """Every auth challenge and status event one endpoint's subscriptions deliver."""
    def __init__(self, endpoint):
        self.lock = threading.Lock()
        self.challenges, self.statuses = [], []
        self.challenged = threading.Event()
        self.auth = live.register_auth_required_callback(endpoint, self.on_challenge)
        self.status = live.register_status_callback(endpoint, self.on_status)

    def on_challenge(self, event):
        with self.lock:
            self.challenges.append(event)
        self.challenged.set()

    def on_status(self, event):
        with self.lock:
            self.statuses.append(event)

    def close(self):
        for subscription in (self.auth, self.status):
            assert subscription.close(5000) == live.Result.OK

    def report(self):
        with self.lock:
            return dict(challenges=[asdict(event) for event in self.challenges],
                        statuses=[dict(endpoint=event.endpoint, uri=event.uri, state=int(event.status.state),
                                       subject=event.status.subject_id) for event in self.statuses])


def install_stack_dump():
    # Loaded by path so the packaged omni.usd.live import keeps the path it is tested with.
    spec = importlib.util.spec_from_file_location(
        'live_stack_dump', Path(__file__).resolve().parents[3] / 'tools/live/stack_dump.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install()


def main():
    install_stack_dump()
    package = sys.argv[1]
    Plug.Registry().RegisterPlugins(str(Path(package) / 'resources/plugInfo.json'))
    runtime = live.initialize(diagnostic_allowed_flags=RESOURCE_NAMES)
    endpoints, observers = {}, {}
    for argument in sys.argv[2:]:
        name, authority, token = argument.split('=')
        # The auth wait outlasts every held challenge; it is a failure-only bound.
        endpoints[name] = runtime.register_endpoint(authority, access_token=token, auth_wait_timeout_ms=60000)
        observers[name] = Observer(endpoints[name])
    host = Host()
    context = host.call(lambda: live.set_publication_context(host.dispatch, max_pending=4))
    scene = dict(stage=None, root=None, layers={})

    def save(uri, timeout_ms):
        # An off-host save returns BUSY while host edits are buffered and
        # schedules the flush job that admits them on the host. A status newer
        # than the one read before the save, with none buffered, or a terminal
        # status, follows their admission or the layer's failure; the save, a
        # drain rather than a mutation, then runs again with the time left.
        layer = host.call(lambda: scene['layers'][uri])
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            seen = live.get_layer_status(uri).revision
            result = live.save_layer(layer, int(max(0.0, deadline - time.monotonic()) * 1000))
            if result != live.Result.BUSY:
                return result
            def admitted():
                status = live.get_layer_status(uri)
                return status.revision > seen and (
                    status.state in (live.State.FAILED, live.State.CLOSING, live.State.CLOSED) or
                    (status.state != live.State.BLOCKED and status.buffered_edit_count == 0))
            until(admitted)

    def state(uri):
        status = live.get_layer_status(uri)
        return dict(fields=host.call(lambda: fields(scene['layers'][uri])), state=int(status.state),
                    result=int(status.result), subject=status.subject_id, pending=status.pending_write_count,
                    buffered=status.buffered_edit_count)

    def endpoint_report(name):
        endpoint = endpoints[name]
        return dict(auth=asdict(live.get_auth_state(endpoint)), status=asdict(live.get_endpoint_status(endpoint)),
                    **observers[name].report())

    print(json.dumps(dict(type='ready', endpoints={name: endpoint.handle for name, endpoint in endpoints.items()})),
          flush=True)
    for line in sys.stdin:
        args = line.split()
        command = args[0]
        if command == 'open-stage':
            uris = args[1:]
            def open_layer(uri):
                # The host opens each live layer on its publication context,
                # through the endpoint that the URI's authority names.
                scene['layers'][uri] = Sdf.Layer.FindOrOpen(uri)
                assert scene['layers'][uri] and scene['layers'][uri].permissionToEdit, uri
            for uri in uris:
                host.call(lambda: open_layer(uri))
                until(lambda: live.get_layer_status(uri).state == live.State.READY)
            def compose():
                # One stage composes both live layers as sublayers of its root.
                scene['root'] = Sdf.Layer.CreateAnonymous('two-services.usda')
                scene['root'].subLayerPaths = uris
                scene['stage'] = Usd.Stage.Open(scene['root'])
                return [layer.identifier for layer in scene['stage'].GetLayerStack(includeSessionLayers=False)]
            result = dict(type='opened', stack=host.call(compose), layers={uri: state(uri) for uri in uris})
        elif command == 'state':
            result = dict(type='state', **state(args[1]))
        elif command == 'set':
            uri, key, value = args[1], args[2], float(args[3])
            def edit():
                with Sdf.ChangeBlock():
                    scene['layers'][uri].pseudoRoot.SetInfo(key, value)
            host.call(edit)
            # Fence: host jobs already on the host queue have run.
            host.call(lambda: None)
            result = dict(type='authored')
        elif command == 'define':
            uri, path, value = args[1], args[2], int(args[3])
            def define():
                layer = scene['layers'][uri]
                with Sdf.ChangeBlock():
                    prim = Sdf.CreatePrimInLayer(layer, path)
                    prim.specifier = Sdf.SpecifierDef
                    attribute = layer.GetAttributeAtPath(path + '.value') or \
                        Sdf.AttributeSpec(prim, 'value', Sdf.ValueTypeNames.Int)
                    attribute.default = value
            host.call(define)
            # Fence: host jobs already on the host queue have run.
            host.call(lambda: None)
            result = dict(type='authored')
        elif command == 'composed':
            def composed():
                # The value the stage composes, and the layer of its strongest opinion.
                attribute = scene['stage'].GetPrimAtPath(args[1]).GetAttribute('value')
                stack = attribute.GetPropertyStack()
                return dict(value=attribute.Get(), layer=stack[0].layer.identifier if stack else None)
            result = dict(type='composed', **host.call(composed))
        elif command == 'save':
            result = dict(type='saved', result=int(save(args[1], int(args[2]))))
        elif command == 'wait':
            uri = args[1]
            expected = {args[index]: float(args[index + 1]) for index in range(2, len(args), 2)}
            until(lambda: host.call(lambda: fields(scene['layers'][uri])) == expected)
            result = dict(type='state', **state(uri))
        elif command == 'wait-state':
            # Failure-only deadline; a FAILED layer returns at once for the assertion.
            uri, expected = args[1], live.State(int(args[2]))
            until(lambda: live.get_layer_status(uri).state in (expected, live.State.FAILED))
            result = dict(type='state', **state(uri))
        elif command == 'wait-status-event':
            # The endpoint's status subscription has delivered the layer's state.
            name, uri, expected = args[1], args[2], int(args[3])
            until(lambda: any(event['uri'] == uri and event['state'] == expected
                              for event in observers[name].report()['statuses']))
            result = dict(type='endpoint', **endpoint_report(name))
        elif command == 'wait-challenge':
            assert observers[args[1]].challenged.wait(10), 'no auth challenge'  # Failure-only watchdog.
            result = dict(type='endpoint', **endpoint_report(args[1]))
        elif command == 'endpoint':
            result = dict(type='endpoint', **endpoint_report(args[1]))
        elif command == 'refresh':
            # Answer the endpoint's challenge with the generation it rejected.
            name, token = args[1], args[2]
            with observers[name].lock:
                generation = observers[name].challenges[-1].rejected_generation
            answer, replaced = live.set_access_token(endpoints[name], generation, token)
            result = dict(type='refreshed', result=int(answer), generation=replaced)
        elif command == 'diagnostics':
            endpoint = endpoints[args[1]] if len(args) > 1 else None
            snapshot = live.copy_diagnostics(endpoint, flags=RESOURCE_NAMES)
            keys = ('endpoint', 'subject_kind', 'subject_id', 'state', 'canonical_resource')
            result = dict(type='diagnostics', endpoint_filter=snapshot['endpoint_filter'],
                          statuses=[{key: status.get(key) for key in keys} for status in snapshot['statuses']],
                          records=[dict(endpoint=record['endpoint'], subject=record['subject_id'],
                                        event=record['event_name'], resource=record.get('canonical_resource'))
                                   for record in snapshot['recent_records']])
        elif command == 'release':
            uri = args[1]
            def release():
                # Drop the sublayer from the stage and every handle to it.
                scene['root'].subLayerPaths.remove(uri)
                scene['layers'].pop(uri)
                gc.collect()
                return [layer.identifier for layer in scene['stage'].GetLayerStack(includeSessionLayers=False)]
            stack = host.call(release)
            until(lambda: layer_state(uri) in (None, live.State.CLOSED))
            result = dict(type='released', stack=stack)
        elif command == 'unregister':
            name = args[1]
            observers.pop(name).close()
            endpoint = endpoints.pop(name)
            def unregistered():
                # A closed layer's actor holds an endpoint lease until the
                # runtime scheduler retires it; the endpoint reports BUSY until then.
                result = endpoint.close(100)
                assert result in (live.Result.OK, live.Result.BUSY, live.Result.TIMEOUT), result
                return result == live.Result.OK
            until(unregistered)
            result = dict(type='unregistered')
        elif command == 'stat':
            answer, _ = live.stat_layer(args[1], timeout_ms=5000)
            result = dict(type='stat', result=int(answer))
        elif command == 'close':
            def drop():
                scene.update(stage=None, root=None, layers={})
                gc.collect()
            host.call(drop)
            def close_context():
                result = context.close(100)
                assert result in (live.Result.OK, live.Result.BUSY, live.Result.TIMEOUT), result
                return result == live.Result.OK
            until(close_context)
            for observer in observers.values():
                observer.close()
            def close_runtime():
                result = runtime.close(5000)
                assert result in (live.Result.OK, live.Result.BUSY), result
                return result == live.Result.OK
            until(close_runtime)
            host.queue.put(None)
            host.thread.join(5)
            assert not host.thread.is_alive()
            print(json.dumps(dict(type='closed')), flush=True)
            return
        else:
            raise AssertionError(command)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
