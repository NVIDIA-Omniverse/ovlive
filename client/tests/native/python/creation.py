# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packaged Python creation consumer; USD lives on one independently progressing host."""
from dataclasses import asdict, FrozenInstanceError
import dis
import gc
import json
from pathlib import Path
import sys
import threading
import weakref

from pxr import Plug, Sdf, Tf
from omni.usd import live
from host import Host, fields, until


def rejected(action, expected):
    try:
        action()
    except live.LiveError as error:
        assert error.result == expected, error
    else:
        raise AssertionError('operation did not reject')


def main():
    authority, session, package = sys.argv[1:4]
    Plug.Registry().RegisterPlugins(str(Path(package) / 'resources/plugInfo.json'))
    runtime = live.initialize(flags=1)  # Operational evidence survives disabled diagnostics.
    endpoint = runtime.register_endpoint(authority, access_token='synthetic-writer')
    host = Host()
    dispatch_completed = threading.Event()
    callback_armed = threading.Event()
    callback_entered = threading.Event()
    callback_release = threading.Event()
    callback_returned = threading.Event()
    callback_results = []
    def dispatch(action):
        if callback_armed.is_set():
            callback_armed.clear()
            assert getattr(live._local, 'runtime', None) is runtime
            callback_entered.set()
            try:
                assert callback_release.wait(5), 'callback release watchdog expired'
                # The calling Python consumer still owns this native task.
                # Blocking here would await our own callback/capture retirement.
                callback_results.append(creation.wait_closed(5000))
                action.cancel()
                action = None
            finally:
                callback_returned.set()
            return
        def run():
            try:
                action()
            finally:
                dispatch_completed.set()
        host.dispatch(run)
    context = host.call(lambda: live.set_publication_context(dispatch, max_pending=1))
    replacement_context = None
    retained_subject = None
    replacement_dispatches = []
    uri = live.make_layer_uri(live.make_session_uri(authority, session), 'created.live')
    creation = None
    waiter = None
    wait_result = []
    # A successful implicit creation hands its layer to the binding; the
    # status callback reports the layer subject CLOSED once the layer goes.
    closed_lock, closed_subjects, closed_changed = threading.Lock(), set(), threading.Event()
    def observe_status(event):
        if event.uri == uri and event.status.state == live.State.CLOSED:
            with closed_lock:
                closed_subjects.add(event.status.subject_id)
            closed_changed.set()
    status_observer = live.register_status_callback(endpoint, observe_status)

    def wait_closed_subject(subject):
        while True:
            with closed_lock:
                if subject in closed_subjects:
                    return
                closed_changed.clear()
            assert closed_changed.wait(5), 'layer subject never reported CLOSED'

    def drop_host_layer():
        # The layer's binding owns an adopted attachment: releasing the last
        # layer reference closes it, and CLOSED follows its worker retirement.
        subject = live.get_layer_status(uri).subject_id
        def drop():
            host.layer = None
            gc.collect()
        host.call(drop)
        wait_closed_subject(subject)
        return subject

    def seed():
        layer = Sdf.Layer.CreateAnonymous('seed.usda')
        layer.startTimeCode = 17.0
        return layer

    def prepare():
        source = seed()
        owner = live.prepare_creation(source, uri, context=context)
        source.startTimeCode = 99.0
        return owner

    def report(kind, result=live.Result.OK):
        outcome = creation.outcome
        assert isinstance(outcome, live.CreationOutcome)
        assert isinstance(outcome.phase, live.CreationPhase) and isinstance(outcome.result, live.Result)
        assert isinstance(outcome.error, live.RequestError)
        return dict(type=kind, result=int(result), outcome=asdict(outcome),
                    owners=len(context.creations), delivery=int(context.delivery_status))

    def close_owner():
        closed = host.call(creation.close)
        if closed == live.Result.BUSY:
            assert creation.wait_closed(5000) == live.Result.OK
            assert host.call(creation.close) == live.Result.OK
        else:
            assert closed == live.Result.OK
        assert not context.creations
        assert creation.wait_closed(0) == live.Result.OK

    print(json.dumps(dict(type='ready')), flush=True)
    for line in sys.stdin:
        args = line.split()
        command = args[0]
        if command == 'retained-prepare':
            assert live.next_retained_creation() is None
            def retained_prepare():
                source = seed()
                subject = live.prepare_retained_creation(source, uri, context=context)
                source.startTimeCode = 99.0
                return subject
            retained_subject = host.call(retained_prepare)
            info = live.next_retained_creation()
            assert isinstance(info, live.RetainedCreationInfo)
            assert (info.subject_id, info.live_uri) == (retained_subject, uri)
            assert live.next_retained_creation(retained_subject) is None
            try:
                info.subject_id = 0
            except FrozenInstanceError:
                pass
            else:
                raise AssertionError('retained identity is mutable')
            assert not context.creations
            result = dict(type='retained-prepared', subject=retained_subject, uri=info.live_uri)
        elif command == 'retained-contracts':
            before = live.next_retained_creation()
            for value in (True, -1, 1 << 64, '1'):
                for action in (lambda: live.next_retained_creation(value),
                               lambda: live.release_retained_creation(value),
                               lambda: context.acquire_retained_creation(value)):
                    try:
                        action()
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('invalid subject accepted')
            rejected(lambda: context.acquire_retained_creation(retained_subject), live.Result.BUSY)
            rejected(lambda: context.prepare_retained_creation(None, uri), live.Result.BUSY)
            assert live.release_retained_creation(retained_subject) == live.Result.BUSY
            assert live.next_retained_creation() == before and not context.creations
            def contracts():
                rejected(lambda: context.acquire_retained_creation(retained_subject + 1), live.Result.NOT_FOUND)
                assert live.release_retained_creation(retained_subject + 1) == live.Result.NOT_FOUND
                rejected(lambda: context.prepare_retained_creation(seed(), uri), live.Result.ALREADY_EXISTS)
                assert live.next_retained_creation() == before and not context.creations
                # Exhaust the Python owner bound without transferring registry ownership.
                blocker = context.prepare_creation(seed(), uri.replace('created.live', 'capacity.live'))
                rejected(lambda: context.acquire_retained_creation(retained_subject), live.Result.RESOURCE_EXHAUSTED)
                assert live.next_retained_creation() == before and context.creations == (blocker,)
                return blocker
            blocker = host.call(contracts)
            code = host.call(blocker.close)
            if code == live.Result.BUSY:
                assert blocker.wait_closed(5000) == live.Result.OK
                assert host.call(blocker.close) == live.Result.OK
            else:
                assert code == live.Result.OK
            assert not context.creations and live.next_retained_creation() == before
            result = dict(type='retained-contracts', subject=retained_subject)
        elif command == 'retained-replace-context':
            replacement_context = host.call(lambda: live.set_publication_context(host.dispatch, max_pending=1))
            host.call(lambda: rejected(lambda: context.prepare_retained_creation(seed(), uri),
                                       live.Result.NOT_INITIALIZED))
            assert live.next_retained_creation().subject_id == retained_subject
            result = dict(type='retained-context-replaced', subject=retained_subject)
        elif command == 'retained-abandon-context':
            old = context
            before = live.next_retained_creation()
            assert before.subject_id == retained_subject and not old.creations
            host.call(old._abandon)
            assert old._done.wait(5), 'abandoned original consumer did not stop'
            old._thread.join(5)
            assert not old._thread.is_alive() and old._wait_running(5)
            assert old.close(3000) == live.Result.OK
            reference = weakref.ref(old)
            def replacement_dispatch(action):
                replacement_dispatches.append(True)
                host.dispatch(action)
            context = host.call(lambda: live.set_publication_context(replacement_dispatch, max_pending=1))
            del old
            gc.collect()
            assert reference() is None, 'original Python publication context remains owned'
            assert live.next_retained_creation() == before
            result = dict(type='retained-context-abandoned', subject=retained_subject, original_destroyed=True)
        elif command in ('retained-acquire', 'retained-interrupt-acquire'):
            if command == 'retained-acquire':
                creation = host.call(lambda: live.acquire_retained_creation(retained_subject, context=context))
            else:
                def interrupted_acquire():
                    code = live.PublicationContext.acquire_retained_creation.__code__
                    instructions = list(dis.get_instructions(code))
                    native_offset = next(op.offset for op in instructions
                                         if op.opname in ('LOAD_METHOD', 'LOAD_ATTR') and
                                         op.argval == 'acquire_retained_creation')
                    result_offset = next(op.offset for op in instructions
                                         if op.opname == 'STORE_FAST' and op.argval == 'result')
                    witnessed = []
                    def trace(frame, event, arg):
                        if frame.f_code == code:
                            frame.f_trace_opcodes = True
                            if event == 'opcode' and frame.f_lasti == native_offset:
                                assert len(context.creations) == 1
                                assert context.creations[0].close() == live.Result.BUSY
                                assert context.close(0) == live.Result.BUSY
                                assert live.next_retained_creation().subject_id == retained_subject
                                witnessed.append('admitting')
                            if event == 'opcode' and frame.f_lasti == result_offset:
                                assert live.next_retained_creation() is None
                                witnessed.append('transferred')
                                raise KeyboardInterrupt('after native retained owner transfer')
                        return trace
                    frame = sys._getframe()
                    frame.f_trace_opcodes = True
                    sys.settrace(trace)
                    try:
                        context.acquire_retained_creation(retained_subject)
                    except KeyboardInterrupt:
                        pass
                    else:
                        raise AssertionError('acquisition interruption did not fire')
                    finally:
                        sys.settrace(None)
                        frame.f_trace_opcodes = False
                    assert witnessed == ['admitting', 'transferred'], witnessed
                host.call(interrupted_acquire)
                assert len(context.creations) == 1
                creation = context.creations[0]
            assert live.next_retained_creation() is None
            assert creation in context.creations and creation.outcome.subject_id == retained_subject
            assert creation.wait_closed(0) == live.Result.INVALID_ARGUMENT
            result = report('retained-acquired')
        elif command == 'retained-release':
            code = host.call(lambda: live.release_retained_creation(retained_subject))
            if code == live.Result.BUSY:
                assert live.next_retained_creation().subject_id == retained_subject
                creation = host.call(lambda: context.acquire_retained_creation(retained_subject))
                assert creation.wait_closed(5000) == live.Result.OK
                assert host.call(creation.close) == live.Result.OK
                creation = None
            else:
                assert code == live.Result.OK
            assert live.next_retained_creation() is None and not context.creations
            assert host.call(lambda: live.release_retained_creation(retained_subject)) == live.Result.NOT_FOUND
            result = dict(type='retained-released', first_result=int(code))
        elif command in ('usd-export', 'usd-create-new'):
            dispatch_completed.clear()
            def invoke():
                source = seed() if command == 'usd-export' else None
                succeeded = False
                try:
                    if source is not None:
                        succeeded = source.Export(uri)
                    else:
                        host.layer = Sdf.Layer.CreateNew(uri)
                        succeeded = host.layer is not None
                except Tf.ErrorException:
                    succeeded = False
                if source is not None:
                    source.startTimeCode = 99.0
                # This call is still on the original outer USD host turn: the
                # workflow stays retained until a later host job adopts it.
                info = live.next_retained_creation()
                assert info is not None and info.live_uri == uri
                if succeeded:
                    rejected(lambda: context.acquire_retained_creation(info.subject_id), live.Result.BUSY)
                if succeeded and command == 'usd-create-new':
                    assert live.get_layer_status(uri).state == live.State.OPENING
                return succeeded, info.subject_id, live.get_layer_status(uri)
            success, retained_subject, status = host.call(invoke)
            if success and command == 'usd-create-new':
                assert dispatch_completed.wait(5), 'queued USD finalizer did not execute'
                host.call(lambda: None)  # Retire task captures after the event inside dispatch.
                status = live.get_layer_status(uri)
            elif success:
                # No host reference holds the Export destination: its adoption
                # job drops the workflow's layer, which closes the subject.
                wait_closed_subject(retained_subject)
            result = dict(type='usd-hook', ok=success, subject=retained_subject,
                          state=int(status.state), result=int(status.result),
                          retained=live.next_retained_creation() is not None)
        elif command == 'drop-created-layer':
            subject = drop_host_layer()
            result = dict(type='created-layer-dropped', subject=subject,
                          retained=live.next_retained_creation() is not None)
        elif command == 'open-created-layer':
            def open_created():
                try:
                    host.layer = Sdf.Layer.FindOrOpen(uri)
                except Tf.ErrorException as error:
                    return str(error)
                return None
            error = host.call(open_created)
            if error is None:
                until(lambda: live.get_layer_status(uri).state == live.State.READY)
                opened = host.call(lambda: fields(host.layer))
            result = dict(type='created-layer-opened', error=error,
                          fields=None if error is not None else opened)
        elif command == 'close-context':
            result = dict(type='context-closed', result=int(context.close(30000)))
        elif command == 'verify-created-layer':
            def verify():
                layer = creation.layer if creation is not None else host.layer
                if host.layer is not None:
                    assert layer == host.layer
                assert layer.identifier == uri and layer.permissionToEdit
                return fields(layer)
            result = dict(type='created-layer', fields=host.call(verify))
        elif command == 'prepare':
            creation = host.call(prepare)
            rejected(lambda: context.prepare_creation(None, uri), live.Result.BUSY)
            for action in (creation.start, creation.prepare_layer, creation.activate, creation.close):
                assert action() == live.Result.BUSY
            rejected(lambda: creation.layer, live.Result.BUSY)
            result = report('prepared')
        elif command == 'interrupt-prepare':
            def interrupted():
                # The public method delegates to the retained-owner handoff.
                # Interrupt exactly after native preparation, before its Result
                # reaches Python; the outer API call remains unchanged below.
                code = live.PublicationContext._prepare_creation.__code__
                offset = next(op.offset for op in dis.get_instructions(code)
                              if op.opname == 'STORE_FAST' and op.argval == 'result')
                fired = []
                def trace(frame, event, arg):
                    if frame.f_code == code:
                        frame.f_trace_opcodes = True
                        if event == 'opcode' and frame.f_lasti == offset:
                            fired.append(True)
                            raise KeyboardInterrupt('deterministic interruption before result delivery')
                    return trace
                # Python 3.12 enables opcode instrumentation when at least one
                # current frame requests it before installing the trace hook.
                frame = sys._getframe()
                frame.f_trace_opcodes = True
                sys.settrace(trace)
                try:
                    prepare()
                except KeyboardInterrupt:
                    pass
                else:
                    raise AssertionError('interrupt did not fire')
                finally:
                    sys.settrace(None)
                    frame.f_trace_opcodes = False
                assert fired == [True]
            host.call(interrupted)
            assert len(context.creations) == 1
            creation = context.creations[0]
            result = report('interrupted')
        elif command == 'contracts':
            def contracts():
                source = seed()
                for text, expected in ((uri + '\x00bad', live.Result.INVALID_ARGUMENT),
                                       ('x' * 65537, live.Result.RESOURCE_EXHAUSTED),
                                       ('é' * 40000, live.Result.RESOURCE_EXHAUSTED)):
                    rejected(lambda: context.prepare_creation(source, text), expected)
                rejected(lambda: context.prepare_creation(None, uri), live.Result.INVALID_ARGUMENT)
                assert not context.creations
            host.call(contracts)
            result = dict(type='contracts')
        elif command in ('retain', 'retain-existing'):
            reference = weakref.ref(creation)
            creation = None
            gc.collect()
            assert reference() is not None
            assert len(context.creations) == 1
            creation = context.creations[0]
            assert creation is reference()
            if command == 'retain':
                host.call(lambda: rejected(lambda: prepare(), live.Result.RESOURCE_EXHAUSTED))
            result = report('retained')
        elif command == 'replace-context':
            old = context
            context = host.call(lambda: live.set_publication_context(host.dispatch, max_pending=1))
            host.call(lambda: rejected(lambda: old.prepare_creation(seed(), uri), live.Result.NOT_INITIALIZED))
            assert not old.creations
            assert old.close(3000) == live.Result.OK
            result = dict(type='replaced')
        elif command == 'start':
            result = report('started', host.call(creation.start))
        elif command == 'wait':
            result = report('waited', creation.wait(int(args[1])))
        elif command == 'wait-background':
            entered = threading.Event()
            def wait():
                entered.set()
                wait_result.append(creation.wait(5000))
            waiter = threading.Thread(target=wait)
            waiter.start()
            assert entered.wait(1)
            # Outcome is thread-neutral but the native active-operation slot
            # rejects overlap. This observes actual admission, not a sleep guess.
            def active():
                try:
                    creation.outcome
                    return False
                except live.LiveError as error:
                    assert error.result == live.Result.BUSY
                    return True
            until(active)
            assert host.call(creation.close) == live.Result.BUSY
            assert context.close(0) == live.Result.BUSY
            result = dict(type='waiting')
        elif command == 'join-wait':
            waiter.join(6)
            assert not waiter.is_alive() and wait_result == [live.Result.OK], wait_result
            result = report('joined')
        elif command == 'prepare-layer':
            result = report('layer-prepared', host.call(creation.prepare_layer))
        elif command == 'activate':
            value = host.call(creation.activate)
            result = report('activated', value)
            if value == live.Result.OK:
                result['fields'] = host.call(lambda: fields(creation.layer))
        elif command == 'hold-close-callback':
            assert not callback_results and not callback_entered.is_set()
            assert host.call(lambda: fields(creation.layer)) == {'startTimeCode': 17.0}
            callback_armed.set()
            result = dict(type='callback-armed')
        elif command == 'close-with-callback-held':
            assert callback_entered.wait(5), 'native publication never entered Python dispatcher'
            assert host.call(creation.close) == live.Result.BUSY
            assert len(context.creations) == 1
            result = dict(type='callback-held-close', result=int(live.Result.BUSY))
        elif command == 'finish-close-callback':
            callback_release.set()
            assert callback_returned.wait(6), 'callback wait_closed did not return'
            assert callback_results == [live.Result.BUSY], callback_results
            assert creation.wait_closed(5000) == live.Result.OK
            assert host.call(creation.close) == live.Result.OK
            assert not context.creations
            result = dict(type='callback-close-finished', callback_result=int(callback_results[0]),
                          owners=len(context.creations))
        elif command == 'throw-dispatch':
            with host.lock:
                host.throw = host.hold = host.hold_once = True
            result = dict(type='throwing')
        elif command == 'failed':
            until(lambda: context.delivery_status != live.Result.OK)
            until(lambda: live.get_layer_status(uri).state == live.State.FAILED)
            with host.lock:
                assert len(host.held) == 1
            host.resume()
            assert host.call(lambda: fields(creation.layer)) == {'startTimeCode': 17.0}
            result = report('failed')
            result['status'] = asdict(live.get_layer_status(uri))
        elif command == 'export':
            assert not Path(args[1]).exists(), 'export destination already exists'
            def export():
                # Only an activated creation has a layer to export.
                assert creation.layer is not None
                # SdfLayer::Export of the failed live layer (design section 13).
                assert creation.layer.Export(args[1])
                return fields(Sdf.Layer.OpenAsAnonymous(args[1]))
            result = dict(type='exported', result=0, fields=host.call(export))
        elif command == 'close-owner':
            close_owner()
            assert host.call(creation.close) == live.Result.OK
            result = dict(type='owner-closed')
        elif command == 'exit-active':
            print(json.dumps(dict(type='exiting')), flush=True)
            return
        elif command == 'close':
            if creation is not None:
                close_owner()
            if host.call(lambda: host.layer is not None and creation is None):
                drop_host_layer()
            host.call(lambda: setattr(host, 'layer', None))
            if replacement_context is not None:
                assert replacement_context.close(3000) == live.Result.OK
            assert context.close(3000) == live.Result.OK
            assert status_observer.close(5000) == live.Result.OK
            def close_runtime():
                # A closed creation's layer actor holds an endpoint lease until
                # the runtime scheduler retires it; the endpoint reports BUSY
                # until then and the runtime owner stays retained for retry.
                result = runtime.close(3000)
                assert result in (live.Result.OK, live.Result.BUSY), result
                return result == live.Result.OK
            until(close_runtime)
            host.queue.put(None)
            host.thread.join(3)
            assert not host.thread.is_alive()
            print(json.dumps(dict(type='closed')), flush=True)
            return
        else:
            raise AssertionError(command)
        print(json.dumps(result), flush=True)
    raise AssertionError('unexpected EOF')


if __name__ == '__main__':
    main()
