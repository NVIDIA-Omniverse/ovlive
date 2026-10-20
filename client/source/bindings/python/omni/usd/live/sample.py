# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless public-API participant. See docs/ovlive-samples.md.

Run with the matching USD Python: python -B -m omni.usd.live.sample --help.
JSON commands on stdin control one owned session/layer. JSON replies go to
stdout; redacted status and diagnostic events go to the new output directory.
The serial host progresses independently while the command thread waits.
"""
import argparse
from concurrent.futures import Future
from collections import deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time

from pxr import Plug, Sdf, Tf
from omni.usd import live


class EventLog:
    """Optional bounded log copies never hold operational callbacks on file I/O."""
    def __init__(self, stream, *, max_records=1024, max_bytes=1024*1024, max_file_bytes=8*1024*1024):
        self.stream = stream
        self.max_records, self.max_bytes, self.max_file_bytes = max_records, max_bytes, max_file_bytes
        self.changed = threading.Condition()
        self.items = deque()
        self.bytes = self.records = self.written = self.dropped = 0
        self.error = None
        self.stopping = False
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, name='sample-event-log', daemon=True)
        self.thread.start()

    def record(self, kind, **data):
        try:
            text = json.dumps(dict(type=kind, **data), sort_keys=True) + '\n'
            size = len(text.encode('utf-8'))
        except Exception:
            with self.changed:
                self.dropped += 1
            return
        with self.changed:
            if self.stopping or self.error or self.records >= self.max_records or size > self.max_bytes - self.bytes:
                self.dropped += 1
                return
            self.items.append((text, size))
            self.bytes += size
            self.records += 1
            self.changed.notify_all()

    def run(self):
        try:
            while True:
                with self.changed:
                    self.changed.wait_for(lambda: self.items or self.stopping)
                    if not self.items:
                        return
                    text, size = self.items.popleft()
                try:
                    if size > self.max_file_bytes - self.written:
                        with self.changed:
                            self.dropped += 1
                    else:
                        if self.stream.write(text) != len(text):
                            raise OSError('short log write')
                        self.stream.flush()
                        self.written += size
                finally:
                    with self.changed:
                        self.bytes -= size
                        self.records -= 1
        except Exception as error:
            with self.changed:
                self.error = type(error).__name__
                self.dropped += 1 + len(self.items)
                self.items.clear()
                self.bytes = self.records = 0
        finally:
            try:
                self.stream.close()
            except Exception as error:
                with self.changed:
                    self.error = type(error).__name__
            self.done.set()

    def close(self):
        with self.changed:
            self.stopping = True
            self.changed.notify_all()
        if not self.done.wait(30):
            raise TimeoutError('log writer retirement')
        self.thread.join()
        return dict(written_bytes=self.written, dropped=self.dropped, error=self.error)


class Host:
    """A serial USD owner, not a live-processing/frame pump."""
    def __init__(self):
        self.jobs = queue.Queue(maxsize=128)
        self.thread = threading.Thread(target=self.run, name='sample-usd-host', daemon=True)
        self.thread.start()

    def run(self):
        while True:
            action = self.jobs.get()
            if action is None:
                return
            action()
            del action  # Retire captured USD ownership on its host.

    def dispatch(self, action):
        self.jobs.put_nowait(action)

    def call(self, action):
        future = Future()
        def invoke():
            try:
                future.set_result(action())
            except BaseException as error:
                future.set_exception(error)
        self.dispatch(invoke)
        return future.result(35)  # Failure-only watchdog; result notification wakes it.

    def close(self):
        self.jobs.put_nowait(None)
        self.thread.join(35)
        if self.thread.is_alive():
            raise TimeoutError('host retirement')


class Participant:
    def __init__(self, authority, output, token, ca_pem, tls_server_name, transport_security):
        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        self.log = EventLog((output / 'events.jsonl').open('x', encoding='utf-8'))
        self.changed = threading.Condition()
        self.statuses = {}
        self.closed = False
        self.creation = self.joined = self.layer = self.created = self.seed = None
        self.session_request = None
        self.uri = None
        self.dirty = self.export_required = self.exported = False
        self.runtime = live.initialize()
        self.endpoint = self.runtime.register_endpoint(authority, access_token=token, ca_pem=ca_pem,
            tls_server_name=tls_server_name, transport_security=transport_security)
        self.host = Host()
        self.context = self.host.call(lambda: live.set_publication_context(self.host.dispatch))
        self.status_observer = live.register_status_callback(self.endpoint, self.status)
        self.auth_observer = live.register_auth_required_callback(
            self.endpoint, lambda event: self.record('auth_required', **asdict(event)))
        self.diagnostics = live.register_diagnostic_callback(
            lambda event: self.record('diagnostic', **asdict(event)), endpoint=self.endpoint)

    def record(self, kind, **data):
        self.log.record(kind, **data)

    def status(self, event):
        with self.changed:
            old = self.statuses.get(event.uri)
            if old is None or event.status.revision > old.revision:
                self.statuses[event.uri] = event.status
            self.changed.notify_all()
        self.record('status', endpoint=event.endpoint, session=event.session, status=asdict(event.status))

    def session_event(self, event):
        # Do not log arbitrary peer payloads or names.
        self.record('session', event_kind=int(event.kind), result=int(event.result))

    @staticmethod
    def checked(result):
        if result != live.Result.OK:
            raise live.LiveError(result)

    @staticmethod
    def closed_when_released(close):
        # A retired layer actor or request can hold its endpoint until the
        # runtime scheduler releases it; the creation retirement wait does not
        # cover that. Close returns BUSY meanwhile and keeps ownership for
        # retry; any other result fails at once. The deadline is a failure-only
        # watchdog.
        deadline = time.monotonic() + 30
        while (result := close()) == live.Result.BUSY and time.monotonic() < deadline:
            time.sleep(.001)
        Participant.checked(result)

    def create(self, base_key, name):
        if self.session_request is not None:
            raise ValueError('one session per participant')
        self.session_request = live.create_session_with_callback(self.endpoint, base_key, name, lambda *_: None)
        self.checked(self.session_request.wait())
        result, created = self.session_outcome()
        self.checked(result)
        self.created = created
        self.record('created_session', session_id=created.session_id, uri=created.uri)
        # CreatedSession deliberately carries no invented creator/time metadata.
        result, metadata = live.list_sessions(self.endpoint, base_key)
        self.checked(result)
        self.joined = live.join_session(created.uri, 'live-headless-sample', self.session_event)
        self.uri = live.make_layer_uri(created.uri, 'sample.live')
        def prepare():
            seed = Sdf.Layer.CreateAnonymous('sample.usda')
            prim = Sdf.PrimSpec(seed, 'Example', Sdf.SpecifierDef, 'Xform')
            Sdf.AttributeSpec(prim, 'value', Sdf.ValueTypeNames.Int).default = 0
            self.seed = seed  # Exported while the creation has no active layer.
            self.creation = self.context.prepare_creation(seed, self.uri)
            self.export_required = True  # Retain every incomplete started workflow.
            return self.creation.start()
        self.checked(self.host.call(prepare))
        return dict(self.resume(), session=asdict(created), metadata=[asdict(item) for item in metadata])

    def session_outcome(self):
        result, created = self.session_request.outcome
        if created is not None:
            self.created = created
        self.record('session_outcome', result=int(result), error=asdict(self.session_request.error),
                    created=asdict(created) if created is not None else None)
        return result, created

    def resume(self):
        if self.creation is None:
            raise ValueError('no layer creation to resume')
        if self.layer is not None:
            return dict(result=0, uri=self.uri)
        self.checked(self.creation.wait())
        outcome = self.creation.outcome
        if outcome.result != live.Result.OK:
            self.export_required = True
            return dict(result=int(outcome.result), uri=self.uri, subject=outcome.subject_id,
                        export_required=True)
        self.checked(self.host.call(self.creation.prepare_layer))
        self.checked(self.host.call(self.creation.activate))
        self.layer = self.host.call(lambda: self.creation.layer)
        self.export_required = False
        return dict(result=0, uri=self.uri)

    def edit(self, value):
        if self.layer is None or self.export_required or type(value) is not int or not -(2**31) <= value < 2**31:
            raise ValueError('edit requires an active layer and int32 value')
        def apply():
            self.exported = False
            self.dirty = True  # Conservative even if result delivery is interrupted.
            with Sdf.ChangeBlock():
                self.layer.GetAttributeAtPath('/Example.value').default = value
        self.host.call(apply)
        return dict(result=0)

    def failed_layer_result(self):
        # The typed cause of a failed layer; INTERNAL_ERROR when no layer reports one.
        if self.uri:
            try:
                status = live.get_layer_status(self.uri)
            except live.LiveError:
                status = None
            if status is not None and status.state == live.State.FAILED and status.result != live.Result.OK:
                return int(status.result)
        return int(live.Result.INTERNAL_ERROR)

    def status_revision(self):
        with self.changed:
            status = self.statuses.get(self.uri)
            return status.revision if status is not None else 0

    def wait_admitted(self, seen, deadline):
        # The BUSY save saw buffered host edits, so each layer status published
        # after their record reports them as buffered until a flush empties the
        # buffer, which publishes a newer revision even when it issues nothing.
        # A status newer than `seen` with none buffered outside BLOCKED, or a
        # terminal status, therefore follows their admission or the layer's
        # failure.
        def admitted():
            status = self.statuses.get(self.uri)
            return status is not None and status.revision > seen and (
                status.state in (live.State.FAILED, live.State.CLOSING, live.State.CLOSED) or
                (status.state != live.State.BLOCKED and status.buffered_edit_count == 0))
        with self.changed:
            if not self.changed.wait_for(admitted, timeout=max(0.0, deadline - time.monotonic())):
                raise TimeoutError('host edit admission')

    def save(self):
        if self.layer is None or self.export_required:
            return dict(result=self.failed_layer_result(), export_required=True)
        # An off-host save returns BUSY while host edits are buffered and
        # schedules the flush that admits them on the host. The status
        # callback reports that admission; then the save retries with the
        # time left. Save off the host: an on-host save flushes inline and
        # only polls, so a bounded blocking drain needs a caller that
        # progresses independently of the host. The deadline is a
        # failure-only watchdog.
        deadline = time.monotonic() + 30
        while True:
            seen = self.status_revision()
            result = live.save_layer(self.layer, int(max(0.0, deadline - time.monotonic()) * 1000))
            if result != live.Result.BUSY:
                break
            self.wait_admitted(seen, deadline)
        status = live.get_layer_status(self.uri)
        if result == live.Result.OK:
            self.dirty = False
        elif status.state == live.State.FAILED:
            self.export_required = True
        return dict(result=int(result), status=asdict(status))

    def export(self, destination):
        # SdfLayer::Export of the local content (design section 13): the live layer
        # once activated, else the caller's own seed. An existing destination
        # is refused so a retry never replaces an earlier export. The refusal
        # is best-effort: a file created between the check and the export is
        # overwritten.
        if self.creation is None:
            raise ValueError('no creation to export')
        path = Path(destination)
        if path.exists():
            return dict(result=int(live.Result.ALREADY_EXISTS), path=destination)
        source = self.layer if self.layer is not None else self.seed
        def write():
            try:
                return source.Export(str(path))
            except Tf.ErrorException:
                return False  # pxr raises posted write errors instead of returning False.
        if not self.host.call(write):
            return dict(result=int(live.Result.IO_ERROR), path=destination)
        self.exported = True
        return dict(result=0, path=destination)

    def execute(self, command):
        operation = command['operation']
        if operation == 'create':
            return self.create(command['base_layer_key'], command.get('name', 'Headless sample'))
        if operation == 'edit':
            return self.edit(command['value'])
        if operation == 'save':
            return self.save()
        if operation == 'resume':
            return self.resume()
        if operation == 'session-outcome':
            if self.session_request is None:
                raise ValueError('no session request')
            self.checked(self.session_request.wait())
            result, created = self.session_outcome()
            return dict(result=int(result), session=asdict(created) if created is not None else None)
        if operation == 'export':
            return self.export(command['path'])
        if operation == 'status':
            return dict(result=0, status=asdict(live.get_layer_status(self.uri)), exported=self.exported)
        if operation == 'refresh':
            token = Path(command['token_file']).read_text(encoding='utf-8').strip()
            if not token:
                raise ValueError('empty token')
            state = live.get_auth_state(self.endpoint)
            result, generation = live.set_access_token(self.endpoint, state.generation, token)
            return dict(result=int(result), generation=generation)
        if operation == 'wait-status':
            expected = live.State(command['state'])
            with self.changed:
                if not self.changed.wait_for(lambda: self.uri in self.statuses and
                                             self.statuses[self.uri].state == expected, timeout=30):
                    raise TimeoutError('status notification')
                return dict(result=0, status=asdict(self.statuses[self.uri]))
        if operation == 'inspect':
            return dict(result=0, value=self.host.call(lambda: self.layer.GetAttributeAtPath('/Example.value').default))
        if operation == 'snapshot':
            path = self.output / command.get('name', 'diagnostics.json')
            if path.parent != self.output or path.name in ('', '.', '..'):
                raise ValueError('snapshot name must be a filename')
            with path.open('x', encoding='utf-8') as output:
                json.dump(live.copy_diagnostics(self.endpoint), output, sort_keys=True)
                output.write('\n')
            return dict(result=0, path=str(path))
        if operation == 'close':
            return self.close()
        raise ValueError('unknown operation')

    def close(self):
        if self.closed:
            return self.final_reply
        if getattr(self, 'retiring_host', False):
            self.host.close()
            self.closed = True
            self.final_reply = dict(result=0, closed=True, logging=self.logging_report)
            return self.final_reply
        self.host.call(lambda: None)  # Retire any command whose caller timed out.
        if self.session_request is not None:
            self.checked(self.session_request.wait())
            self.session_outcome()  # Retain eventual identity/error before runtime retirement.
        if self.layer is not None:
            status = live.get_layer_status(self.uri)
            if status.state == live.State.FAILED:
                self.export_required = True
        if self.export_required and not self.exported:
            return dict(result=self.failed_layer_result(), retained='export')
        if self.dirty and not self.exported:
            result = self.save()
            if result['result'] != 0:
                return result  # Keep every owner and the progressing host alive.
        if self.creation is not None:
            result = self.host.call(self.creation.close)
            if result == live.Result.BUSY:
                self.checked(self.creation.wait_closed())
                result = self.host.call(self.creation.close)
            self.checked(result)
            self.creation = None
        self.host.call(lambda: (setattr(self, 'layer', None), setattr(self, 'seed', None)))
        if self.joined is not None:
            self.checked(self.joined.leave())
            self.joined = None
        for name in ('status_observer', 'auth_observer', 'diagnostics'):
            subscription = getattr(self, name)
            if subscription is not None:
                self.checked(subscription.close())
                setattr(self, name, None)
        self.logging_report = self.log.close()
        self.checked(self.context.close())
        self.closed_when_released(self.endpoint.close)
        self.closed_when_released(self.runtime.close)
        self.retiring_host = True
        return self.close()


def reply(value):
    try:
        print(json.dumps(value, sort_keys=True), flush=True)
        return True
    except OSError:
        # Losing the reply channel is not authorization to drop live ownership.
        sys.stdout = open(os.devnull, 'w')
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ca-bundle', type=Path, help='PEM trust roots file')
    parser.add_argument('--tls-server-name', default='', help='DNS name of the TLS server')
    parser.add_argument('--transport-security', choices=('default', 'tls', 'plaintext'), default='default')
    args = parser.parse_args()
    token = os.environ.pop('LIVE_SAMPLE_ACCESS_TOKEN', '')
    if not token:
        parser.error('LIVE_SAMPLE_ACCESS_TOKEN is required')
    package = Path(__file__).resolve().parents[4]
    Plug.Registry().RegisterPlugins(str(package / 'resources/plugInfo.json'))
    ca_bytes = b''
    if args.ca_bundle:
        try:
            with args.ca_bundle.open('rb') as bundle:
                ca_bytes = bundle.read(1024 * 1024 + 1)
        except OSError:
            parser.error('cannot read CA bundle')
    if len(ca_bytes) > 1024 * 1024:
        parser.error('CA bundle exceeds 1 MiB')
    try:
        ca_pem = ca_bytes.decode('utf-8')
    except UnicodeError:
        parser.error('CA bundle is not UTF-8 PEM')
    participant = Participant(args.authority, args.output.absolute(), token, ca_pem,
        args.tls_server_name, {'default': live.TransportSecurity.DEFAULT,
                               'tls': live.TransportSecurity.TLS,
                               'plaintext': live.TransportSecurity.PLAINTEXT}[args.transport_security])
    token = None
    retry = threading.Event()
    # EOF or an export failure must not turn into implicit data discard. The
    # retry event (SIGUSR1 on Linux, CTRL_BREAK_EVENT on Windows) asks the
    # retained participant to retry its local export after an operator fixes
    # disk space/permissions or removes a collision.
    if sys.platform == 'win32':
        # Python runs signal handlers only on a main thread that is executing
        # bytecode, never during the wait below; Windows runs a console control
        # handler on a thread of its own.
        import ctypes
        from ctypes import wintypes

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
        def on_control(event):
            if event != 1:  # CTRL_BREAK_EVENT
                return False
            retry.set()
            return True
        if not ctypes.windll.kernel32.SetConsoleCtrlHandler(on_control, True):
            raise ctypes.WinError()
        retry_signal = 'CTRL_BREAK_EVENT'
    else:
        signal.signal(signal.SIGUSR1, lambda *_: retry.set())
        retry_signal = 'SIGUSR1'
    connected = reply(dict(type='ready'))
    while connected:
        try:
            line = sys.stdin.readline()
        except (OSError, KeyboardInterrupt):
            break
        if not line:
            break
        try:
            command = json.loads(line)
            result = participant.execute(command)
        except KeyboardInterrupt:
            break
        except Exception as error:
            # Never echo input/token bytes or arbitrary exception/server text.
            result = dict(result=int(error.result) if isinstance(error, live.LiveError) else -1,
                          error=type(error).__name__)
        connected = reply(dict(type='reply', **result))
        if participant.closed:
            return 0
    while True:
        retry.clear()
        try:
            result = participant.close()
            if not participant.closed and participant.creation is not None:
                if participant.layer is None and participant.creation.outcome.result in (live.Result.OK, live.Result.BUSY):
                    try:
                        participant.resume()  # Continue the same owner; never start twice.
                    except live.LiveError:
                        pass  # Terminal bootstrap failure still needs export below.
                    result = participant.close()
                if participant.closed:
                    reply(dict(type='reply', **result))
                    return 0
                participant.checked(participant.export(str(participant.output / 'eof-export.usda'))['result'])
                result = participant.close()
            if participant.closed:
                reply(dict(type='reply', **result))
                return 0
        except Exception:
            pass  # Keep the owners and host alive; no arbitrary error text.
        reply(dict(type='retained', retry_signal=retry_signal, output=str(participant.output)))
        retry.wait()  # Operator event, not polling or a success timeout.


if __name__ == '__main__':
    raise SystemExit(main())
