# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh Python/pxr consumer of the actual packaged USD host bridge."""
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

from pxr import Plug, Sdf, Tf
from omni.usd import live


def fields(layer):
    return {key: layer.pseudoRoot.GetInfo(key) for key in layer.pseudoRoot.ListInfoKeys()}


def until(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "Python host progress deadline"
        time.sleep(.001)


class Host:
    def __init__(self):
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.held = []
        self.hold = False
        self.hold_once = False
        self.throw = False
        self.layer = self.other = None
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
        with self.lock:
            if self.hold:
                self.held.append(action)
                if self.hold_once:
                    self.hold = self.hold_once = False
            else:
                self.queue.put(action)
            if self.throw:
                raise RuntimeError("deterministic queued-then-throw dispatch")

    def resume(self):
        with self.lock:
            self.hold = self.hold_once = False
            held, self.held = self.held, []
        for action in held:
            self.queue.put(action)


def install_stack_dump():
    # Loaded by path so the packaged omni.usd.live import keeps the path it is tested with.
    spec = importlib.util.spec_from_file_location(
        'live_stack_dump', Path(__file__).resolve().parents[3] / 'tools/live/stack_dump.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install()


def main():
    install_stack_dump()
    authority, session, package = sys.argv[1:4]
    options = dict(option.partition('=')[::2] for option in sys.argv[4:])
    assert set(options) <= {'no-history', 'max-pending'}, options
    disable_history = 'no-history' in options
    Plug.Registry().RegisterPlugins(str(Path(package) / 'resources/plugInfo.json'))
    runtime = live.initialize(flags=1) if disable_history else live.initialize()
    endpoint = runtime.register_endpoint(authority, access_token='synthetic-writer')
    host = Host()
    # One outstanding charged host job unless the test asks for more; a layer's
    # own publication job is not charged.
    max_pending = int(options.get('max-pending', 1))
    context = host.call(lambda: live.set_publication_context(host.dispatch, max_pending=max_pending))
    notice_entered, notice_release, notice_retired = threading.Event(), threading.Event(), threading.Event()
    notice_close = []
    uri = live.make_layer_uri(live.make_session_uri(authority, session), 'root.live')
    other_uri = live.make_layer_uri(live.make_session_uri(authority, session), 'other.live')
    observed_lock, observed_event = threading.Lock(), threading.Event()
    observed_failures = []
    def observe_status(event):
        assert isinstance(event, live.StatusEvent) and isinstance(event.status, live.Status)
        assert isinstance(event.status.result, live.Result) and isinstance(event.status.state, live.State)
        if event.uri == uri and event.status.state == live.State.FAILED:
            with observed_lock:
                observed_failures.append(event.status)
            observed_event.set()
    observer = live.register_status_callback(endpoint, observe_status) if disable_history else None
    def open_layer():
        host.layer = Sdf.Layer.FindOrOpen(uri)
        assert host.layer and host.layer.permissionToEdit
    host.call(open_layer)
    until(lambda: live.get_layer_status(uri).state == live.State.READY)
    original_subject = live.get_layer_status(uri).subject_id

    def report(kind):
        actual = host.call(lambda: fields(host.layer))
        status = live.get_layer_status(uri)
        assert isinstance(status, live.Status) and isinstance(status.result, live.Result) and isinstance(status.state, live.State)
        return dict(type=kind, fields=actual, result=int(status.result), state=int(status.state),
                    pending=status.pending_write_count, delivery=int(context.delivery_status),
                    subject=status.subject_id, incident=status.incident_id, status=asdict(status))

    def save(layer, timeout_ms, target=uri):
        # An off-host save returns BUSY while host edits are buffered and
        # schedules the flush job that admits them on the host (design section 11).
        # Each layer status published after their record reports them as
        # buffered until a flush empties the buffer, which publishes a newer
        # revision even when it issues nothing. A status newer than the one
        # read before the save, with none buffered outside BLOCKED, or a
        # terminal status, therefore follows their admission or the layer's
        # failure; the save, a drain rather than a mutation, then runs again
        # with the time left. The deadline only bounds failure.
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            seen = live.get_layer_status(target).revision
            result = live.save_layer(layer, int(max(0.0, deadline - time.monotonic()) * 1000))
            if result != live.Result.BUSY:
                return result
            def admitted():
                status = live.get_layer_status(target)
                return status.revision > seen and (
                    status.state in (live.State.FAILED, live.State.CLOSING, live.State.CLOSED) or
                    (status.state != live.State.BLOCKED and status.buffered_edit_count == 0))
            until(admitted)

    print(json.dumps(report('ready')), flush=True)
    for line in sys.stdin:
        args = line.split()
        command = args[0]
        if command == 'state':
            result = report('state')
        elif command == 'failed':
            until(lambda: live.get_layer_status(uri).state == live.State.FAILED)
            result = report('failed')
        elif command == 'wait-pending':
            # `set` returns before its flush job has passed through the context's
            # consumer thread to the host, so its edit can still be buffered.
            # Failure-only deadline; a FAILED layer returns at once for the assertion.
            expected = int(args[1])
            until(lambda: live.get_layer_status(uri).pending_write_count >= expected or
                  live.get_layer_status(uri).state == live.State.FAILED)
            result = report('state')
        elif command == 'observed-failure':
            assert observer is not None and observed_event.wait(5), 'operational status callback did not report failure'
            with observed_lock:
                observed = observed_failures[-1]
            assert observed.subject_id == original_subject
            assert observer.delivery_status == live.Result.OK
            result = dict(type='observed-failure', status=asdict(observed))
        elif command == 'diagnostics':
            snapshot = live.copy_diagnostics(endpoint)
            result = dict(type='diagnostics', records=snapshot['recent_records'])
        elif command == 'export':
            path = args[1]
            assert not Path(path).exists(), 'export destination already exists'
            before = live.get_layer_status(uri)
            def export():
                # SdfLayer::Export of the failed live layer (design section 13).
                assert host.layer.Export(path)
                return fields(Sdf.Layer.OpenAsAnonymous(path))
            exported = host.call(export)
            result = dict(type='exported', fields=exported, status_unchanged=before == live.get_layer_status(uri))
        elif command == 'open-other':
            def open_other():
                host.other = Sdf.Layer.FindOrOpen(other_uri)
                assert host.other and host.other.permissionToEdit
            host.call(open_other)
            until(lambda: live.get_layer_status(other_uri).state == live.State.READY)
            result = dict(type='other-opened', fields=host.call(lambda: fields(host.other)))
        elif command == 'other-state':
            actual = host.call(lambda: fields(host.other))
            status = live.get_layer_status(other_uri)
            result = dict(type='other-state', fields=actual, result=int(status.result),
                          state=int(status.state), buffered=status.buffered_edit_count,
                          delivery=int(context.delivery_status))
        elif command == 'other-set':
            def edit():
                with Sdf.ChangeBlock():
                    host.other.pseudoRoot.SetInfo(args[1], float(args[2]))
            host.call(edit)
            host.call(lambda: None)  # Fence: the refused or queued flush dispatch has returned.
            result = dict(type='authored', result=0)
        elif command == 'other-wait-state':
            # Failure-only deadline; a FAILED layer returns at once for the assertion.
            expected = live.State(int(args[1]))
            until(lambda: live.get_layer_status(other_uri).state in (expected, live.State.FAILED))
            status = live.get_layer_status(other_uri)
            result = dict(type='other-state', state=int(status.state), result=int(status.result),
                          buffered=status.buffered_edit_count, delivery=int(context.delivery_status))
        elif command == 'other-wait-buffered':
            # Failure-only deadline; a FAILED layer returns at once for the assertion.
            expected = int(args[1])
            until(lambda: live.get_layer_status(other_uri).buffered_edit_count == expected or
                  live.get_layer_status(other_uri).state == live.State.FAILED)
            status = live.get_layer_status(other_uri)
            result = dict(type='other-state', state=int(status.state), result=int(status.result),
                          buffered=status.buffered_edit_count, delivery=int(context.delivery_status))
        elif command == 'other-save':
            layer = host.call(lambda: host.other)
            result = dict(type='saved', result=int(save(layer, int(args[1]), other_uri)))
            layer = None
        elif command == 'set':
            def edit():
                with Sdf.ChangeBlock():
                    host.layer.pseudoRoot.SetInfo(args[1], float(args[2]))
            host.call(edit)
            # Fence: host jobs already on the host queue have run. The flush job
            # the change block queued reaches this queue asynchronously through
            # the context's consumer thread, so `save` waits for its admission.
            host.call(lambda: None)
            result = dict(type='authored', result=0)
        elif command == 'save':
            layer = host.call(lambda: host.layer)
            result = dict(type='saved', result=int(save(layer, int(args[1]))))
            layer = None
        elif command == 'save-host':
            # A save on the host admits buffered edits in the calling thread,
            # without a flush job; zero polls and reports pending work as BUSY.
            result = dict(type='saved', result=int(host.call(lambda: live.save_layer(host.layer, 0))))
        elif command == 'wait':
            expected = {args[index]: float(args[index + 1]) for index in range(2, len(args), 2)}
            until(lambda: host.call(lambda: fields(host.layer) == expected))
            result = report('ready')
        elif command == 'contracts':
            def contracts():
                ordinary = Sdf.Layer.CreateAnonymous()
                assert live.save_layer(ordinary, 0) == live.Result.INVALID_ARGUMENT
                for name in ('begin_edit', 'acquire_read_view', 'ReadView', 'Edit'):
                    assert not hasattr(live, name), name
                for name in ('Editor', 'ReadView', 'acquire_read_view', 'allocate_edit', 'prepare_edit'):
                    assert not hasattr(live._live, name), name
                assert not hasattr(context, 'edits') and not hasattr(context, 'begin_edit')
                for name in ('export_layer_recovery', 'RecoveryCheck', 'RecoveryRetirement'):
                    assert not hasattr(live, name), name
                assert not hasattr(live._live, 'export_layer_recovery')
                for name in ('check_recovery', 'begin_recovery_retirement', 'recovery_retirement'):
                    assert not hasattr(live.Creation, name), name
                    assert not hasattr(live._live.Creation, name), name
                assert host.layer.permissionToEdit
                return fields(host.layer)
            result = dict(type='checked', fields=host.call(contracts))
        elif command in ('hold', 'hold-one'):
            with host.lock:
                host.hold = True
                host.hold_once = command == 'hold-one'
                host.throw = len(args) > 1 and args[1] == 'throw'
            result = dict(type='holding')
        elif command == 'held':
            with host.lock:
                result = dict(type='held', count=len(host.held), delivery=int(context.delivery_status))
        elif command == 'resume':
            host.resume()
            result = dict(type='resumed')
        elif command == 'close-context':
            result = dict(type='context-close', result=int(context.close(0)))
        elif command == 'exit-active':
            print(json.dumps(dict(type='exit-active')), flush=True)
            return  # atexit must disarm/join bridge work despite a live layer.
        elif command == 'hold-notice':
            def register():
                def receive(notice, sender):
                    notice_close.append(context.close(0))
                    notice_entered.set()
                    notice_release.wait()
                    notice_retired.set()
                host.notice_callback = receive  # Tf.Notice stores a weak Python callback.
                host.notice = Tf.Notice.Register(Sdf.Notice.LayersDidChangeSentPerLayer, receive, host.layer)
            host.call(register)
            result = dict(type='notice-registered')
        elif command == 'notice-entered':
            result = dict(type='notice-entered', entered=notice_entered.is_set())
        elif command == 'teardown-running':
            assert notice_entered.wait(5)
            assert notice_close == [live.Result.BUSY], notice_close
            assert context._running == 1
            joining = threading.Event()
            real_wait = context._wait_running
            def wait_running(timeout=None):
                if live._stopping:  # Teardown's bounded join, or the close it issues.
                    joining.set()
                return real_wait(timeout)
            context._wait_running = wait_running
            def unblock():
                assert joining.wait(5), 'teardown never waited for running publication'
                assert not notice_retired.is_set()
                notice_release.set()
            helper = threading.Thread(target=unblock, daemon=True)
            helper.start()
            live._teardown()
            assert joining.is_set() and notice_retired.is_set() and context._running == 0
            helper.join(3)
            assert not helper.is_alive() and not context._thread.is_alive()
            host.call(lambda: host.notice.Revoke())
            print(json.dumps(dict(type='running-publication-retired')), flush=True)
            return
        elif command == 'close':
            def drop():
                host.layer = host.other = None
                gc.collect()
            host.call(drop)
            def close_context():
                result = context.close(100)
                assert result in (live.Result.OK, live.Result.BUSY, live.Result.TIMEOUT), result
                return result == live.Result.OK
            until(close_context)
            if observer is not None:
                assert observer.close(5000) == live.Result.OK
            def close_runtime():
                # A closed layer's actor holds an endpoint lease until the
                # runtime scheduler retires it; the endpoint reports BUSY
                # until then and the runtime owner stays retained for retry.
                result = runtime.close(5000)
                assert result in (live.Result.OK, live.Result.BUSY), result
                return result == live.Result.OK
            until(close_runtime)
            result = dict(type='closed')
            host.queue.put(None)
            host.thread.join(5)
            assert not host.thread.is_alive()
            print(json.dumps(result), flush=True)
            return
        else:
            raise AssertionError(command)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
