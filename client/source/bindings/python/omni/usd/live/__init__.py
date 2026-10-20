# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned Live Services requests and sessions; USD authoring remains in pxr.Sdf.

Callbacks run on bounded Python workers, never native transport threads. Marshal
USD access to your host context. Close owners explicitly; unsuccessful closes
retain ownership for retry. Diagnostic loss is separate from operational status.
"""
from __future__ import annotations

import atexit
import asyncio
from collections import deque
from dataclasses import dataclass, fields, replace
from enum import IntEnum
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Callable

if sys.platform == "win32":
    # Windows searches neither PATH nor a package's own directories for an extension module's DLL dependencies.
    # Importing pxr.Sdf loads the USD and oneTBB DLLs, which the loader then reuses by name for _live's imports,
    # and the plugin DLL that _live imports is in the package's bin directory.
    from pxr import Sdf  # noqa: F401
    _PACKAGE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
    _DLL_DIRECTORY = os.add_dll_directory(os.path.join(_PACKAGE_ROOT, "bin"))

from . import _live

# The longest native wait (about 49.7 days). close, abandonment and
# interrupt_wait end it; a TIMEOUT only restarts it.
_WAIT_FOREVER_MS = 0xffffffff


class Result(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    INVALID_HANDLE = 2
    NOT_INITIALIZED = 3
    NOT_FOUND = 4
    ALREADY_EXISTS = 5
    UNAUTHENTICATED = 6
    PERMISSION_DENIED = 7
    UNAVAILABLE = 8
    TIMEOUT = 9
    CANCELLED = 10
    MALFORMED_STREAM = 11
    UNSUPPORTED = 12
    RESOURCE_EXHAUSTED = 13
    BUSY = 14
    STALE_TOKEN_GENERATION = 15
    OBJECT_REPLACED = 16
    # 17 is reserved and never assigned.
    BUFFER_TOO_SMALL = 18
    OUTCOME_UNKNOWN = 19
    INTERNAL_ERROR = 20
    IO_ERROR = 21


class TransportSecurity(IntEnum):
    DEFAULT = 0
    TLS = 1
    PLAINTEXT = 2


class State(IntEnum):
    OPENING = 1
    READY = 2
    RECONNECTING = 3
    AUTH_WAIT = 4
    BLOCKED = 5
    # 6 is reserved and never assigned.
    FAILED = 7
    CLOSING = 8
    CLOSED = 9


class SessionEventKind(IntEnum):
    CONNECTION = 1
    JOINED = 2
    LEFT = 3
    MESSAGE = 4


class Severity(IntEnum):
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4


class DiagnosticFieldKind(IntEnum):
    UINT64 = 1
    INT64 = 2
    STRING = 3
    BOOLEAN = 4


@dataclass(frozen=True)
class SessionEvent:
    kind: SessionEventKind
    result: Result
    connection_id: str | None
    user_name: str | None
    app_name: str | None
    bytes: bytes


@dataclass(frozen=True)
class CreatedSession:
    session_id: str
    uri: str


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    uri: str
    name: str
    created_by: str
    created_time_ms: int


@dataclass(frozen=True)
class LayerInfo:
    created_time_ms: int
    modified_time_ms: int


@dataclass(frozen=True)
class AuthState:
    generation: int
    expires_at_ms: int
    challenge_id: int
    state: int
    reason: int


@dataclass(frozen=True)
class AuthRequired:
    endpoint: int
    rejected_generation: int
    challenge_id: int
    reason: int
    grpc_status_present: bool
    grpc_status: int


@dataclass(frozen=True)
class StatusEvent:
    endpoint: int
    session: int
    uri: str
    status: Status


@dataclass(frozen=True)
class DiagnosticField:
    name: str
    kind: DiagnosticFieldKind
    value: int | bool | str


@dataclass(frozen=True)
class DiagnosticRecord:
    schema_version: int
    event_code: int
    event_name: str
    severity: Severity
    timestamp_ms: int
    elapsed_ms: int
    runtime_instance_id: int
    sequence: int
    endpoint: int
    subject_id: int
    incident_id: int
    fields: tuple[DiagnosticField, ...]


@dataclass(frozen=True)
class Capabilities:
    delegate_version: int
    diagnostic_schema_version: int
    autonomous_progress: bool


@dataclass(frozen=True)
class UriParts:
    authority: str
    session_id: str
    layer_name: str
    canonical_uri: str
    is_layer: bool


@dataclass(frozen=True)
class RequestError:
    result: Result
    operation: int = 0
    grpc_status_present: bool = False
    grpc_status: int = 0
    request_id: int = 0
    attempt_id: int = 0
    incident_id: int = 0
    detail: str = ""


@dataclass(frozen=True)
class Status:
    subject_kind: int
    state: State
    subject_id: int
    revision: int
    changed_at_ms: int
    result: Result
    operation: int
    grpc_status_present: bool
    grpc_status: int
    recovery_action: int
    progress_known: bool
    incident_id: int
    pending_write_count: int
    pending_bytes: int
    has_unconfirmed_edits: bool
    blocked_layers: int
    failed_layers: int
    blocked_sessions: int
    failed_sessions: int
    buffered_edit_count: int = 0
    read_sequence: int = 0
    applied_sequence: int = 0


class LiveError(RuntimeError):
    def __init__(self, result: int):
        self.result = Result(result)
        super().__init__(self.result.name)


class RequestTimeout(LiveError):
    """The local blocking wait expired; request remains owned and uncanceled."""
    def __init__(self, request: Request):
        super().__init__(Result.TIMEOUT)
        self.request = request


def _record(cls, value):
    result = {field.name: value[field.name] for field in fields(cls) if field.name in value}
    if "result" in result:
        result["result"] = Result(result["result"])
    if cls is Status:
        result["state"] = State(result["state"])
    if cls is SessionEvent:
        result["kind"] = SessionEventKind(result["kind"])
    if cls is DiagnosticField:
        result["kind"] = DiagnosticFieldKind(result["kind"])
    if cls is DiagnosticRecord:
        result["severity"] = Severity(result["severity"])
    for name in ("grpc_status_present", "progress_known", "has_unconfirmed_edits", "autonomous_progress", "is_layer"):
        if name in result:
            result[name] = bool(result[name])
    return cls(**result)


def _query(pair):
    result, value = pair
    if result != Result.OK:
        raise LiveError(result)
    return value


def _timeout(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xffffffff:
        raise ValueError("timeout_ms must be an integer in [0, 4294967295]")
    return value


def _remaining(deadline):
    return max(0, min(0xffffffff, int((deadline - time.monotonic()) * 1000)))


_local = threading.local()
_owners_lock = threading.RLock()
_owners: list[Runtime] = []
_stopping = False
_log = logging.getLogger(__name__)


def _log_failure(message):
    try:
        _log.exception(message)
    except BaseException:
        pass  # A host logging handler must not strand accepted work.


class Request:
    """A finite request plus its Python delivery obligation.

    wait/cancel return a wait result; outcome contains the operation result.
    TIMEOUT does not discard the request. A callback must not wait on itself.
    OUTCOME_UNKNOWN may mean a confirmed Create whose identity could not be
    delivered to Python. Never retry that mutation; native_status retains the
    underlying native operation result separately from delivery failure.
    """
    def __init__(self, runtime, operation, native, submission, callback, internal):
        self._runtime = runtime
        self._operation = operation
        self._native = native
        self._callback = callback
        self._internal = internal
        self._cancelled = threading.Event()
        self._done = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.RLock()
        self._result = Result(submission)
        self._native_status = Result.BUSY
        self._native_status_known = False
        self._error = RequestError(self._result)
        self._fallback_errors = tuple(RequestError(result) for result in Result)
        self._payload = None
        self._closed = False

    @property
    def native_status(self) -> Result:
        with self._lock:
            return self._native_status

    @property
    def status(self) -> Result:
        with self._lock:
            if not self._ready.is_set():
                return Result.BUSY
            return self._result

    @property
    def error(self) -> RequestError:
        with self._lock:
            if not self._ready.is_set():
                return RequestError(Result.BUSY)
            return self._error

    @property
    def outcome(self):
        with self._lock:
            if not self._ready.is_set():
                raise LiveError(Result.BUSY)
            return self._outcome()

    def _outcome(self):
        if self._operation == "list_sessions":
            return self._result, self._payload if self._payload is not None else []
        if self._operation in ("create_session", "stat_layer"):
            return self._result, self._payload
        return self._result

    def wait(self, timeout_ms: int = 30000) -> Result:
        _timeout(timeout_ms)
        if getattr(_local, "runtime", None) is not None and not self._done.is_set():
            return Result.BUSY
        return Result.OK if self._done.wait(timeout_ms / 1000) else Result.TIMEOUT

    def cancel(self, timeout_ms: int = 30000) -> Result:
        self._cancelled.set()
        return self.wait(timeout_ms)

    def close(self, timeout_ms: int = 30000) -> Result:
        result = self.cancel(timeout_ms) if not self._done.is_set() else Result.OK
        if result == Result.OK:
            self._closed = True
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        result = self.close()
        if result != Result.OK and exc_type is None:
            raise LiveError(result)

    def _run(self):
        native = self._native
        try:
            self._result = Result(self._result)
            if native is None:
                self._native_status = self._result
                self._native_status_known = True
            if native is not None:
                while True:
                    result = native.cancel(25) if self._cancelled.is_set() else native.wait(25)
                    if result in (Result.TIMEOUT, Result.BUSY):
                        continue
                    if result != Result.OK:
                        self._result = Result(result)
                    else:
                        raw_query, raw_status = native.native_status()
                        if raw_query == Result.OK:
                            self._native_status = Result(raw_status)
                            self._native_status_known = True
                        query_result, status = native.status()
                        self._result = Result(status if query_result == Result.OK else query_result)
                        error_result, error = native.error()
                        if error_result == Result.OK:
                            self._error = _record(RequestError, error)
                        payload_result, payload = native.payload()
                        if self._result == Result.OK:
                            self._result = Result(payload_result)
                            if (payload_result != Result.OK and self._operation == "create_session"
                                    and self._native_status == Result.OK):
                                self._result = Result.OUTCOME_UNKNOWN
                            if payload_result == Result.OK:
                                if self._operation == "list_sessions":
                                    self._payload = [_record(SessionInfo, item) for item in payload]
                                elif self._operation == "create_session":
                                    self._payload = _record(CreatedSession, payload)
                                elif self._operation == "stat_layer":
                                    self._payload = _record(LayerInfo, payload)
                    break
        except BaseException:
            self._bridge_failure()
            _log_failure("Live request completion failed")
        finally:
            # A cancellation may suppress the native callback. Native quiescence,
            # not payload delivery, is the release barrier.
            if native is not None:
                while native.release() in (Result.BUSY, Result.TIMEOUT):
                    native.cancel(25)
                self._native = None
            if self._error.result != self._result:
                self._align_error()
        self._finish()

    def _align_error(self):
        try:
            self._error = replace(self._error, result=self._result)
        except BaseException:
            self._error = self._fallback_errors[int(self._result)]

    def _bridge_failure(self):
        if self._native_status_known and self._native_status != Result.OK:
            self._result = self._native_status
        elif self._result not in (Result.OK, Result.BUSY):
            pass  # Keep an already observed native/bridge failure.
        elif self._operation == "create_session":
            self._result = Result.OUTCOME_UNKNOWN
        elif self._operation in ("delete_session", "delete_layer", "send_message"):
            # A known successful delete needs no payload. Without a terminal
            # native result, delivery failure cannot establish mutation failure.
            self._result = (Result.OK if self._native_status == Result.OK
                            else Result.OUTCOME_UNKNOWN)
        else:
            self._result = Result.INTERNAL_ERROR
        self._align_error()

    def _finish(self):
        try:
            self._ready.set()
            callback = self._callback
            if callback is not None and (self._internal or
                    (not self._runtime._disarmed and not self._cancelled.is_set())):
                value = self._outcome()
                callback(*value) if isinstance(value, tuple) else callback(value)
        except BaseException:
            _log_failure("Live request callback failed")
        finally:
            self._callback = None
            self._done.set()


class Runtime:
    """One initialization reference with its own bounded Python task queue."""
    def __init__(self, native, workers, capacity, max_sessions, max_subscriptions, max_host_contexts):
        self._native = native
        self._condition = threading.Condition(threading.RLock())
        self._queue = deque()
        self._requests = set()
        self._endpoints = set()
        self._sessions = set()
        self._max_sessions = max_sessions
        self._subscriptions = set()
        self._max_subscriptions = max_subscriptions
        self._hosts = set()
        self._max_host_contexts = max_host_contexts
        self._capacity = capacity
        self._reserved = 0
        self._closing = False
        self._closed = False
        self._disarmed = False
        self._threads = [threading.Thread(target=self._worker, name="omni.usd.live", daemon=True)
                         for _ in range(workers)]
        started = []
        try:
            for thread in self._threads:
                thread.start()
                started.append(thread)
        except BaseException:
            with self._condition:
                self._closed = True
                self._condition.notify_all()
            for thread in started:
                thread.join()
            raise

    def _worker(self):
        _local.runtime = self
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._closed)
                if not self._queue:
                    return
                request = self._queue.popleft()
            _local.request = request
            try:
                request._run()
            except BaseException:
                request._bridge_failure()
                request._payload = None
                # Native owners defer cleanup if an exceptional Python path
                # interrupted explicit release; never abandon callback storage.
                request._native = None
                _log_failure("Live Python worker failed")
            finally:
                try:
                    if not request._done.is_set():
                        request._finish()
                finally:
                    _local.request = None
                    with self._condition:
                        self._requests.discard(request)
                        self._reserved -= 1
                        self._condition.notify_all()
                    del request  # Idle workers must not retain completed payloads.

    def _submit(self, operation, args, callback, internal):
        with self._condition:
            if self._closing or self._closed or _stopping:
                raise LiveError(Result.NOT_INITIALIZED)
            if self._reserved >= self._capacity:
                raise LiveError(Result.RESOURCE_EXHAUSTED)
            self._reserved += 1
            request = None
            try:
                # Reserve Python ownership before the native mutation can be
                # admitted. The worker cannot see it until this lock is released.
                request = Request(self, operation, None, Result.BUSY, callback, internal)
                self._requests.add(request)
                self._queue.append(request)
                result, native = getattr(_live, operation)(*args)
                request._native = native
                request._result = result
            except BaseException:
                if request is not None:
                    self._requests.discard(request)
                    if request in self._queue:
                        self._queue.remove(request)
                self._reserved -= 1
                self._condition.notify_all()
                raise
            self._condition.notify()
            return request

    def close(self, timeout_ms: int = 30000) -> Result:
        deadline = time.monotonic() + _timeout(timeout_ms) / 1000
        with self._condition:
            if self._closed:
                return Result.OK
            self._closing = True
            self._disarmed = True
            for request in self._requests:
                request._cancelled.set()
            sessions = tuple(self._sessions)
            subscriptions = tuple(self._subscriptions)
            hosts = tuple(self._hosts)
        if getattr(_local, "runtime", None) is not None:
            for session in sessions:
                session.leave(0)
            for subscription in subscriptions:
                subscription.close(0)
            for host in hosts:
                host.close(0)
            return Result.BUSY
        for session in sessions:
            result = session.leave(_remaining(deadline))
            if result != Result.OK:
                return result
        with self._condition:
            while self._reserved:
                left = deadline - time.monotonic()
                if left <= 0:
                    return Result.TIMEOUT
                self._condition.wait(left)
            subscriptions = tuple(self._subscriptions)
            endpoints = tuple(self._endpoints)
        for subscription in subscriptions:
            result = subscription.close(_remaining(deadline))
            if result != Result.OK:
                return result
        for host in hosts:
            result = host.close(_remaining(deadline))
            if result != Result.OK:
                return result
        for endpoint in endpoints:
            result = endpoint.close(_remaining(deadline))
            if result != Result.OK:
                return result
        with self._condition:
            if self._closed:
                return Result.OK
            result = Result(self._native.close(_remaining(deadline)))
            if result != Result.OK:
                return result
            self._closed = True
            self._condition.notify_all()
        for thread in self._threads:
            thread.join()
        with _owners_lock:
            if self in _owners:
                _owners.remove(self)
        return Result.OK

    def register_endpoint(self, authority: str, **options) -> Endpoint:
        with self._condition:
            if self._closing or self._closed:
                raise LiveError(Result.NOT_INITIALIZED)
            native = _query(_live.register_endpoint(dict(options, authority=authority)))
            endpoint = Endpoint(self, native)
            self._endpoints.add(endpoint)
            return endpoint

    def join_session(self, uri: str, app_name: str, callback: Callable[[SessionEvent], Any], *,
                     max_events: int = 256, max_bytes: int = 1048576) -> Session:
        if not callable(callback):
            raise TypeError("callback must be callable")
        if (type(max_events) is not int or not 1 <= max_events <= 4096 or
                type(max_bytes) is not int or not 1 <= max_bytes <= 16 * 1024 * 1024):
            raise ValueError("require 1 <= max_events <= 4096 and 1 <= max_bytes <= 16777216")
        session = None
        started = False
        try:
            with self._condition:
                if self._closing or self._closed or _stopping:
                    raise LiveError(Result.NOT_INITIALIZED)
                if len(self._sessions) >= self._max_sessions:
                    raise LiveError(Result.RESOURCE_EXHAUSTED)
                # Allocate the callback owner, bounded slot and consumer thread
                # before the native join can create a service participant.
                session = Session(self, callback)
                self._sessions.add(session)
                session._thread.start()
                started = True
                result, native = _live.join_session(uri, app_name, max_events, max_bytes)
                session._native = native
                if result != Result.OK:
                    raise LiveError(result)
                session._admitted.set()
                return session
        except BaseException:
            if session is not None:
                session._closing = True
                session._admitted.set()
                if started:
                    session._thread.join()
                with self._condition:
                    self._sessions.discard(session)
            raise

    def _subscribe(self, kind, endpoint, callback, on_error, options=None, max_records=256, max_bytes=1048576):
        if not callable(callback) or (on_error is not None and not callable(on_error)):
            raise TypeError("callback and optional on_error must be callable")
        if (type(max_records) is not int or not 1 <= max_records <= 4096 or
                type(max_bytes) is not int or not 1 <= max_bytes <= 16 * 1024 * 1024):
            raise ValueError("invalid subscription queue limits")
        subscription = None
        started = False
        try:
            with self._condition:
                if self._closing or self._closed or _stopping:
                    raise LiveError(Result.NOT_INITIALIZED)
                if len(self._subscriptions) >= self._max_subscriptions:
                    raise LiveError(Result.RESOURCE_EXHAUSTED)
                subscription = Subscription(self, kind, endpoint, callback, on_error)
                self._subscriptions.add(subscription)
                subscription._thread.start()
                started = True
                native_endpoint = endpoint._native if endpoint is not None else None
                if kind == "diagnostic":
                    result, native = _live.register_diagnostic_callback(native_endpoint, options, max_records, max_bytes)
                elif kind == "status":
                    result, native = _live.register_status_callback(native_endpoint)
                else:
                    result, native = _live.register_auth_required_callback(native_endpoint)
                subscription._native = native
                if result != Result.OK:
                    raise LiveError(result)
                subscription._admitted.set()
                return subscription
        except BaseException:
            if subscription is not None:
                subscription._closing = True
                subscription._admitted.set()
                if started:
                    subscription._thread.join()
                with self._condition:
                    self._subscriptions.discard(subscription)
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        result = self.close()
        if result != Result.OK and exc_type is None:
            raise LiveError(result)


class Endpoint:
    """One registration reference; equal handles do not imply equal ownership."""
    def __init__(self, runtime, native):
        self._runtime = runtime
        self._native = native
        self._lock = threading.RLock()
        self._closed = False

    @property
    def handle(self) -> int:
        return self._native.handle

    def close(self, timeout_ms: int = 30000) -> Result:
        with self._lock:
            if self._closed:
                return Result.OK
            result = Result(self._native.close(_timeout(timeout_ms)))
            if result == Result.OK:
                self._closed = True
        if result == Result.OK:
            with self._runtime._condition:
                self._runtime._endpoints.discard(self)
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        result = self.close()
        if result != Result.OK and exc_type is None:
            raise LiveError(result)


class Session:
    """An independent joined participant and its ordered Python event stream.

    Send success means local queue acceptance, not peer receipt. Leave suppresses
    queued Python delivery and stops transport promptly. TIMEOUT retains this
    Python owner for retry; the native handle may already have been released
    while an executing Python callback is still retiring.
    """
    def __init__(self, runtime, callback):
        self._runtime = runtime
        self._native = None
        self._callback = callback
        self._condition = threading.Condition(threading.RLock())
        self._leave_lock = threading.Lock()
        self._admitted = threading.Event()
        self._done = threading.Event()
        self._closing = False
        self._closed = False
        self._native_released = False
        self._delivery_result = Result.OK
        self._resource_event = SessionEvent(SessionEventKind.CONNECTION, Result.RESOURCE_EXHAUSTED,
                                            None, None, None, b"")
        self._internal_event = SessionEvent(SessionEventKind.CONNECTION, Result.INTERNAL_ERROR,
                                            None, None, None, b"")
        self._thread = threading.Thread(target=self._consume, name="omni.usd.live.session", daemon=True)

    @property
    def handle(self) -> int:
        return self._native.handle

    @property
    def delivery_status(self) -> Result:
        if self._delivery_result != Result.OK:
            return self._delivery_result
        return Result(self._native.delivery_status())

    @property
    def status(self) -> Status:
        return get_session_status(self)

    def _emit(self, event):
        with self._condition:
            if self._closing or self._runtime._disarmed:
                return
            callback = self._callback
        if callback is not None:
            try:
                callback(event)
            except BaseException:
                _log_failure("Live session callback failed")

    def _consume(self):
        _local.runtime = self._runtime
        _local.session = self
        try:
            self._admitted.wait()
            if self._native is None or self._closing:
                return
            while not self._closing:
                # Native condition-variable wait: transport remains autonomous.
                # interrupt_wait/leave wakes this wait without a host frame pump.
                result, value = self._native.next_event(_WAIT_FOREVER_MS)
                if result == Result.TIMEOUT:
                    continue
                if result in (Result.CANCELLED, Result.INVALID_HANDLE):
                    break
                if result != Result.OK:
                    raise LiveError(result)
                event = _record(SessionEvent, value)
                self._emit(event)
                del event, value
        except BaseException as error:
            result = Result.RESOURCE_EXHAUSTED if isinstance(error, MemoryError) else Result.INTERNAL_ERROR
            self._delivery_result = result
            _log_failure("Live session event delivery failed")
            if self._native is not None:
                # Stop before reporting the terminal bridge failure. No silently
                # lost event is followed by apparently healthy delivery/sends.
                try:
                    self._native.interrupt_wait()
                    if self._leave_lock.acquire(False):
                        try:
                            if not self._native_released and self._native.leave(0) == Result.OK:
                                self._native_released = True
                        finally:
                            self._leave_lock.release()
                except BaseException:
                    _log_failure("Live session transport cleanup failed")
            self._emit(self._resource_event if result == Result.RESOURCE_EXHAUSTED else self._internal_event)
        finally:
            self._callback = None
            self._done.set()
            _local.session = None
            _local.runtime = None

    def leave(self, timeout_ms: int = 30000) -> Result:
        deadline = time.monotonic() + _timeout(timeout_ms) / 1000
        with self._condition:
            if self._closed:
                return Result.OK
            self._closing = True
        worker = getattr(_local, "runtime", None) is not None
        acquired = self._leave_lock.acquire(timeout=0 if worker else _remaining(deadline) / 1000)
        if not acquired:
            return Result.BUSY if worker else Result.TIMEOUT
        try:
            if self._closed:
                return Result.OK
            if not self._native_released:
                self._native.interrupt_wait()
                result = Result(self._native.leave(0 if worker else _remaining(deadline)))
                if result == Result.OK:
                    self._native_released = True
                elif result not in (Result.BUSY, Result.TIMEOUT):
                    return result
                elif not worker:
                    return result
            if worker:
                return Result.BUSY
            if not self._done.wait(_remaining(deadline) / 1000):
                return Result.TIMEOUT
            self._thread.join()
            self._closed = True
        finally:
            self._leave_lock.release()
        with self._runtime._condition:
            self._runtime._sessions.discard(self)
        return Result.OK

    def close(self, timeout_ms: int = 30000) -> Result:
        return self.leave(timeout_ms)

    def send_message(self, data: bytes, *, timeout_ms: int = 30000) -> Result:
        return _blocking("send_message", (self, data), timeout_ms)

    def send_message_async(self, data: bytes):
        return _async("send_message", (self, data))

    def send_message_with_callback(self, data: bytes, callback: Callable[[Result], Any]) -> Request:
        return _start("send_message", (self, data), callback)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        result = self.leave()
        if result != Result.OK and exc_type is None:
            raise LiveError(result)


class Subscription:
    """A bounded callback owner; status ACK follows successful Python delivery.

    Operational callback/materialization failure stops dispatch without ACK or
    unregister, preserving retained native status until explicit close. Diagnostic
    dropped_records counts only this bridge's queue drops; native sink/history
    losses remain visible in copy_diagnostics().
    """
    def __init__(self, runtime, kind, endpoint, callback, on_error):
        self._runtime, self._kind, self._endpoint = runtime, kind, endpoint
        self._callback, self._on_error = callback, on_error
        self._native = None
        self._admitted = threading.Event()
        self._done = threading.Event()
        self._condition = threading.Condition(threading.RLock())
        self._close_lock = threading.Lock()
        self._closing = self._closed = self._native_released = False
        self._delivery_result = Result.OK
        self._thread = threading.Thread(target=self._consume, name="omni.usd.live." + kind, daemon=True)

    @property
    def handle(self) -> int:
        return self._native.handle

    @property
    def delivery_status(self) -> Result:
        return (self._delivery_result if self._delivery_result != Result.OK
                else Result(self._native.delivery_status()))

    @property
    def dropped_records(self) -> int:
        return self._native.dropped_records

    def _consume(self):
        _local.runtime = self._runtime
        _local.subscription = self
        callback = event = value = on_error = None
        try:
            self._admitted.wait()
            if self._native is None or self._closing:
                return
            while not self._closing:
                # close() ends this wait: interrupt_wait wakes auth and
                # diagnostic waits, and unregistration cancels a status read.
                result, value = self._native.next_event(_WAIT_FOREVER_MS)
                if result == Result.TIMEOUT:
                    continue
                if self._closing:
                    break
                if result != Result.OK:
                    raise LiveError(result)
                if self._kind == "status":
                    event = _record(StatusEvent, dict(value, status=_record(Status, value["status"])))
                    subject, revision = event.status.subject_id, event.status.revision
                elif self._kind == "auth":
                    event = _record(AuthRequired, value)
                    # Recheck immediately before invoking Python, including when
                    # an older handler was held while credentials were replaced.
                    # The C subscription lease is independent of Endpoint owner.
                    if not self._native.auth_current(event.rejected_generation, event.challenge_id):
                        del event, value
                        continue
                else:
                    owned_fields = tuple(_record(DiagnosticField, field) for field in value["fields"])
                    event = _record(DiagnosticRecord, dict(value, fields=owned_fields))
                    del owned_fields
                with self._condition:
                    callback = None if self._closing or self._runtime._disarmed else self._callback
                if callback is None:
                    del event, value
                    break
                if self._kind == "diagnostic":
                    try:
                        callback(event)
                    except BaseException:
                        _log_failure("Live diagnostic callback failed")
                else:
                    callback(event)  # Failure must retain the selected status unacknowledged.
                del callback, event, value
                if self._kind == "status" and not self._closing:
                    result = self._native.acknowledge(subject, revision)
                    if result != Result.OK and not self._closing:
                        raise LiveError(result)
        except BaseException as error:
            if not self._closing:
                self._delivery_result = (Result.RESOURCE_EXHAUSTED if isinstance(error, MemoryError)
                                         else error.result if isinstance(error, LiveError) else Result.INTERNAL_ERROR)
                _log_failure("Live subscription delivery stopped")
                on_error = self._on_error
                if on_error is not None and not self._runtime._disarmed:
                    try:
                        on_error(self._delivery_result)
                    except BaseException:
                        _log_failure("Live subscription error callback failed")
                # In particular, no ACK/unregister/interrupt of retained status.
        finally:
            callback = event = value = on_error = None
            self._callback = None
            self._on_error = None
            self._done.set()
            _local.subscription = None
            _local.runtime = None

    def close(self, timeout_ms: int = 30000) -> Result:
        deadline = time.monotonic() + _timeout(timeout_ms) / 1000
        with self._condition:
            if self._closed:
                return Result.OK
            self._closing = True
        worker = getattr(_local, "runtime", None) is not None
        if not self._close_lock.acquire(timeout=0 if worker else _remaining(deadline) / 1000):
            return Result.BUSY if worker else Result.TIMEOUT
        try:
            if self._closed:
                return Result.OK
            if not self._native_released:
                self._native.interrupt_wait()
                result = Result(self._native.close(0 if worker else _remaining(deadline)))
                if result == Result.OK:
                    self._native_released = True
                elif result not in (Result.BUSY, Result.TIMEOUT):
                    return result
                elif not worker:
                    return result
            if worker:
                return Result.BUSY
            if not self._done.wait(_remaining(deadline) / 1000):
                return Result.TIMEOUT
            self._thread.join()
            self._closed = True
        finally:
            self._close_lock.release()
        with self._runtime._condition:
            self._runtime._subscriptions.discard(self)
        return Result.OK

    def unregister(self, timeout_ms: int = 30000) -> Result:
        return self.close(timeout_ms)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        result = self.close()
        if result != Result.OK and exc_type is None:
            raise LiveError(result)


def initialize(*, python_workers: int = 4, python_max_pending: int = 256,
               python_max_sessions: int = 64, python_max_subscriptions: int = 64,
               python_max_host_contexts: int = 8, **limits) -> Runtime:
    if (type(python_workers) is not int or type(python_max_pending) is not int or
            not 1 <= python_workers <= 64 or not python_workers <= python_max_pending <= 65536):
        raise ValueError("require 1 <= python_workers <= 64 and workers <= python_max_pending <= 65536")
    if type(python_max_sessions) is not int or not 1 <= python_max_sessions <= 256:
        raise ValueError("python_max_sessions must be an integer in [1, 256]")
    if type(python_max_subscriptions) is not int or not 1 <= python_max_subscriptions <= 256:
        raise ValueError("python_max_subscriptions must be an integer in [1, 256]")
    if type(python_max_host_contexts) is not int or not 1 <= python_max_host_contexts <= 64:
        raise ValueError("python_max_host_contexts must be an integer in [1, 64]")
    with _owners_lock:
        if _stopping:
            raise LiveError(Result.NOT_INITIALIZED)
        native = _query(_live.initialize(limits))
        owner = Runtime(native, python_workers, python_max_pending, python_max_sessions, python_max_subscriptions,
                        python_max_host_contexts)
        _owners.append(owner)
        return owner


def _current() -> Runtime:
    with _owners_lock:
        for owner in reversed(_owners):
            if not owner._closing:
                return owner
    raise LiveError(Result.NOT_INITIALIZED)


def shutdown(timeout_ms: int = 30000) -> Result:
    """Close one Python initialization owner, retaining it on failed cleanup."""
    with _owners_lock:
        if not _owners:
            return Result.NOT_INITIALIZED
        owner = _owners[-1]
    return owner.close(timeout_ms)


def register_endpoint(authority: str, **options) -> Endpoint:
    return _current().register_endpoint(authority, **options)


def join_session(uri: str, app_name: str, callback: Callable[[SessionEvent], Any], *,
                 max_events: int = 256, max_bytes: int = 1048576) -> Session:
    return _current().join_session(uri, app_name, callback, max_events=max_events, max_bytes=max_bytes)


def unregister_endpoint(endpoint: Endpoint, timeout_ms: int = 30000) -> Result:
    return endpoint.close(timeout_ms)


def register_auth_required_callback(endpoint: Endpoint, callback: Callable[[AuthRequired], Any], *,
                                    on_error: Callable[[Result], Any] | None = None) -> Subscription:
    if not isinstance(endpoint, Endpoint):
        raise TypeError("endpoint must be an owned Endpoint")
    return endpoint._runtime._subscribe("auth", endpoint, callback, on_error)


def register_status_callback(endpoint: Endpoint | None, callback: Callable[[StatusEvent], Any], *,
                             on_error: Callable[[Result], Any] | None = None) -> Subscription:
    if endpoint is not None and not isinstance(endpoint, Endpoint):
        raise TypeError("endpoint must be an owned Endpoint or None")
    owner = endpoint._runtime if endpoint is not None else _current()
    return owner._subscribe("status", endpoint, callback, on_error)


def register_diagnostic_callback(callback: Callable[[DiagnosticRecord], Any], *, endpoint: Endpoint | None = None,
                                 min_severity: int = Severity.INFO, flags: int = 0, event_codes=(),
                                 max_records: int = 256, max_bytes: int = 1048576,
                                 on_error: Callable[[Result], Any] | None = None) -> Subscription:
    if endpoint is not None and not isinstance(endpoint, Endpoint):
        raise TypeError("endpoint must be an owned Endpoint or None")
    owner = endpoint._runtime if endpoint is not None else _current()
    return owner._subscribe("diagnostic", endpoint, callback, on_error,
                            dict(min_severity=min_severity, flags=flags, event_codes=event_codes), max_records, max_bytes)


def get_auth_state(endpoint: Endpoint) -> AuthState:
    return _record(AuthState, _query(endpoint._native.auth_state()))


def set_access_token(endpoint: Endpoint, expected_generation: int, token: str, expires_at_ms: int = 0) -> tuple[Result, int]:
    result, generation = endpoint._native.set_access_token(expected_generation, token, expires_at_ms)
    return Result(result), generation


def fail_auth_refresh(endpoint: Endpoint, challenge_id: int) -> Result:
    return Result(endpoint._native.fail_auth_refresh(challenge_id))


def get_endpoint_status(endpoint: Endpoint) -> Status:
    return _record(Status, _query(endpoint._native.status()))


def get_layer_status(uri: str) -> Status:
    return _record(Status, _query(_live.get_layer_status(uri)))


def get_session_status(session: Session) -> Status:
    if not isinstance(session, Session):
        raise TypeError("session must be an owned Session")
    return _record(Status, _query(session._native.status()))


def _start(operation, args, callback=None, internal=False):
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    first = args[0]
    if isinstance(first, Session):
        with first._condition:
            if first._closed:
                raise LiveError(Result.INVALID_HANDLE)
            if first._closing:
                raise LiveError(Result.CANCELLED)
            delivery = first.delivery_status
            if delivery != Result.OK:
                raise LiveError(delivery)
    owner = first._runtime if isinstance(first, (Endpoint, Session)) else _current()
    native_args = (first._native,) + args[1:] if isinstance(first, (Endpoint, Session)) else args
    return owner._submit(operation, native_args, callback, internal)


def _failure(operation, result):
    if operation == "list_sessions":
        return Result(result), []
    if operation in ("create_session", "stat_layer"):
        return Result(result), None
    return Result(result)


def _blocking(operation, args, timeout_ms):
    _timeout(timeout_ms)
    if getattr(_local, "runtime", None) is not None:
        return _failure(operation, Result.BUSY)
    try:
        request = _start(operation, args)
    except LiveError as error:
        return _failure(operation, error.result)
    result = request.wait(timeout_ms)
    if result == Result.TIMEOUT:
        raise RequestTimeout(request)
    if result != Result.OK:
        return _failure(operation, result)
    return request.outcome


def _async(operation, args):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    holder = []
    def deliver(*values):
        value = values if len(values) != 1 else values[0]
        def settle():
            if not future.done():
                future.set_result(value)
        try:
            loop.call_soon_threadsafe(settle)
        except RuntimeError:
            pass  # Closed loop: worker still releases the native request.
    try:
        request = _start(operation, args, deliver, internal=True)
        holder.append(request)
    except LiveError as error:
        future.set_result(_failure(operation, error.result))
    def finished(completed):
        if completed.cancelled() and holder:
            holder[0]._cancelled.set()
        holder.clear()
    future.add_done_callback(finished)
    return future


def list_sessions(endpoint: Endpoint, base_layer_key: str, *, timeout_ms: int = 30000):
    return _blocking("list_sessions", (endpoint, base_layer_key), timeout_ms)


def list_sessions_async(endpoint: Endpoint, base_layer_key: str):
    return _async("list_sessions", (endpoint, base_layer_key))


def list_sessions_with_callback(endpoint: Endpoint, base_layer_key: str, callback: Callable) -> Request:
    return _start("list_sessions", (endpoint, base_layer_key), callback)


def create_session(endpoint: Endpoint, base_layer_key: str, session_name: str, *, timeout_ms: int = 30000):
    return _blocking("create_session", (endpoint, base_layer_key, session_name), timeout_ms)


def create_session_async(endpoint: Endpoint, base_layer_key: str, session_name: str):
    return _async("create_session", (endpoint, base_layer_key, session_name))


def create_session_with_callback(endpoint: Endpoint, base_layer_key: str, session_name: str, callback: Callable) -> Request:
    return _start("create_session", (endpoint, base_layer_key, session_name), callback)


def stat_layer(uri: str, *, timeout_ms: int = 30000):
    return _blocking("stat_layer", (uri,), timeout_ms)


def stat_layer_async(uri: str):
    return _async("stat_layer", (uri,))


def stat_layer_with_callback(uri: str, callback: Callable) -> Request:
    return _start("stat_layer", (uri,), callback)


def delete_layer(uri: str, *, timeout_ms: int = 30000):
    return _blocking("delete_layer", (uri,), timeout_ms)


def delete_layer_async(uri: str):
    return _async("delete_layer", (uri,))


def delete_layer_with_callback(uri: str, callback: Callable) -> Request:
    return _start("delete_layer", (uri,), callback)


def delete_session(uri: str, *, timeout_ms: int = 30000):
    return _blocking("delete_session", (uri,), timeout_ms)


def delete_session_async(uri: str):
    return _async("delete_session", (uri,))


def delete_session_with_callback(uri: str, callback: Callable) -> Request:
    return _start("delete_session", (uri,), callback)


def get_abi_version() -> int:
    return _query(_live.get_abi_version())


def get_capabilities() -> Capabilities:
    return _record(Capabilities, _query(_live.get_capabilities()))


def get_result_string(result: int) -> str:
    return _query(_live.get_result_string(result))


def make_session_uri(authority: str, session_id: str) -> str:
    return _query(_live.make_session_uri(authority, session_id))


def make_layer_uri(session_uri: str, layer_name: str) -> str:
    return _query(_live.make_layer_uri(session_uri, layer_name))


def combine_with_base_uri(base_uri: str, relative_layer_name: str) -> str:
    return _query(_live.combine_with_base_uri(base_uri, relative_layer_name))


def parse_uri(uri: str) -> UriParts:
    return _record(UriParts, _query(_live.parse_uri(uri)))


def copy_diagnostics(endpoint: Endpoint | None = None, **options) -> dict[str, Any]:
    return json.loads(_query(_live.copy_diagnostics(endpoint._native if endpoint else None, options)))


def export_diagnostics_json(endpoint: Endpoint | None = None, **options) -> str:
    return _query(_live.copy_diagnostics(endpoint._native if endpoint else None, options))


def configure_jitter_reduction(endpoint: Endpoint, delay_constant_ms: int,
                               delay_multiple: int, delay_maximum_ms: int) -> Result:
    return Result(_live.configure_jitter_reduction(endpoint._native, dict(delay_constant_ms=delay_constant_ms,
                  delay_multiple=delay_multiple, delay_maximum_ms=delay_maximum_ms)))


def wait_for_pending_updates(timeout_ms: int = 30000) -> Result:
    return Result(_live.wait_for_pending_updates(_timeout(timeout_ms)))


_teardown_watchdog_s = 30.0  # Failure-only bound on interpreter-exit waits.
# Failure-only bound on the teardown runtime close. It bounds only the native
# completion-queue poller join, runs no Python and is independent of the Python
# teardown watchdog.
_native_release_s = 30.0


def _joined(thread, timeout):
    thread.join(timeout)
    return not thread.is_alive()


def _released(result):
    return Result(result) in (Result.OK, Result.INVALID_HANDLE)


def _release_stream(stream, lock, release):
    if not lock.acquire(blocking=False):
        return False  # Another thread is closing it; its result is not waited on.
    try:
        if stream._native is not None and not stream._native_released:
            stream._native_released = _released(release(stream._native))
        return stream._native is None or stream._native_released
    finally:
        lock.release()


def _release_native(owner):
    """Close the native side of every owner that Runtime.close left open.

    Streams and endpoints close with zero timeouts; the runtime close waits only
    for native quiescence. No Python callback and no Result wait runs here
    (design section 19). Returns the names of native owners that stay owned.
    """
    busy = []
    with owner._condition:
        if owner._closed:
            return busy
        sessions = tuple(owner._sessions)
        subscriptions = tuple(owner._subscriptions)
        endpoints = tuple(owner._endpoints)
    for session in sessions:
        if not _release_stream(session, session._leave_lock, lambda native: native.leave(0)):
            busy.append("native session")
    for subscription in subscriptions:
        if not _release_stream(subscription, subscription._close_lock, lambda native: native.close(0)):
            busy.append("native subscription")
    for endpoint in endpoints:
        if not endpoint._lock.acquire(blocking=False):
            busy.append("native endpoint")
            continue
        try:
            if not endpoint._closed:
                if _released(endpoint._native.close(0)):
                    endpoint._closed = True
                else:
                    busy.append("native endpoint")
        finally:
            endpoint._lock.release()
    with owner._condition:
        if not owner._closed and not _released(owner._native.close(int(_native_release_s * 1000))):
            busy.append("native runtime")
    return busy


def _teardown():
    global _stopping
    with _owners_lock:
        if _stopping and all(owner._closed for owner in _owners):
            return  # atexit repeats an explicit teardown.
    deadline = time.monotonic() + _teardown_watchdog_s

    def left():
        return max(0.0, deadline - time.monotonic())

    abandoned = []
    with _owners_lock:
        _stopping = True
        owners = tuple(_owners)
    for owner in owners:
        with owner._condition:
            owner._disarmed = True
            for request in owner._requests:
                request._callback = None
                request._cancelled.set()
            sessions = tuple(owner._sessions)
            subscriptions = tuple(owner._subscriptions)
            hosts = tuple(owner._hosts)
        for host in hosts:
            host._abandon()
        for session in sessions:
            session.leave(0)
        for subscription in subscriptions:
            subscription.close(0)
    # Drain this module's Python execution while the interpreter/GIL still
    # exist. Every wait ends as soon as its work retires. A user callback that is
    # already executing cannot be revoked: at the watchdog, teardown records it
    # and lets the interpreter exit instead of hanging the host (design section 19).
    for owner in owners:
        owner.close(_remaining(deadline))
        with owner._condition:
            if not owner._condition.wait_for(lambda: not owner._reserved, left()):
                abandoned.append("request callback")
            sessions = tuple(owner._sessions)
            subscriptions = tuple(owner._subscriptions)
            hosts = tuple(owner._hosts)
        for host in hosts:
            if not (host._done.wait(left()) and _joined(host._thread, left()) and host._wait_running(left())):
                abandoned.append("host context")
        for session in sessions:
            if not (session._done.wait(left()) and _joined(session._thread, left())):
                abandoned.append("session")
        for subscription in subscriptions:
            if not (subscription._done.wait(left()) and _joined(subscription._thread, left())):
                abandoned.append("subscription")
        owner.close(_remaining(deadline))
        abandoned.extend(_release_native(owner))
        with owner._condition:
            owner._closed = True
            owner._condition.notify_all()
        for thread in owner._threads:
            # A worker runs user code only while a request is reserved; an idle
            # worker that misses an exhausted deadline is not reported.
            if not _joined(thread, left()) and owner._reserved:
                abandoned.append("worker")
    if abandoned:
        _log.error("Live Services teardown abandoned unfinished owners (%s) after %.3g s; "
                   "the interpreter exits without them", ", ".join(sorted(set(abandoned))), _teardown_watchdog_s)


from ._host import (PublicationContext, Creation, CreationPhase, CreationOutcome,
                    RetainedCreationInfo, next_retained_creation, release_retained_creation,
                    prepare_retained_creation, acquire_retained_creation,
                    set_publication_context, prepare_creation, save_layer)

atexit.register(_teardown)
