# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serial USD host dispatch and creation owners."""
import sys
import threading
import time
from dataclasses import dataclass
from enum import IntEnum

_api = sys.modules[__package__]


class PublicationContext:
    """One serial host. Register on that host and dispatch jobs back to it.

    The dispatcher must enqueue without waiting and execute jobs after the
    current USD call returns, outside Sdf.ChangeBlock. BUSY close stops future
    opens but keeps existing attachments and their dispatcher progressing.
    """
    def __init__(self, runtime, dispatch, on_error, max_pending=64):
        self._runtime = runtime
        self._dispatch = dispatch
        self._on_error = on_error
        self._creator = threading.get_ident()
        self._native = None
        self._admitted = threading.Event()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._close_lock = threading.Lock()
        self._activity = threading.Condition(threading.RLock())
        self._running = 0
        self._closed = False
        self._retiring = False
        self._closing = False
        self._disarmed = False
        self._failure = _api.Result.OK
        self._creations = set()
        self._max_creations = max_pending
        self._thread = threading.Thread(target=self._consume, name="omni.usd.live.host", daemon=True)

    @property
    def delivery_status(self):
        if self._failure != _api.Result.OK:
            return self._failure
        return _api.Result(self._native.delivery_status()) if self._native else _api.Result.NOT_INITIALIZED

    def _failed(self, result):
        self._failure = _api.Result(result)
        self._disarmed = True
        self._stop.set()
        if self._native is not None:
            self._native.abandon()
        if self._on_error is not None and not self._runtime._disarmed:
            try:
                self._on_error(self._failure)
            except BaseException:
                _api._log_failure("Live USD host delivery error handler failed")

    def _consume(self):
        _api._local.runtime = self._runtime
        self._admitted.wait()
        try:
            if self._native is None:
                return
            while not self._stop.is_set():
                # close, abandon and interrupt_wait end this wait.
                result, native_task = self._native.next_task(_api._WAIT_FOREVER_MS)
                if result == _api.Result.TIMEOUT:
                    continue
                # CANCELLED means the native context is closed, abandoned or
                # interrupted. close() sets _closing before the native close
                # that marks it closed, and _abandon, _failed and close set
                # _stop before abandon or interrupt_wait. A BUSY native close
                # leaves the context open, so its wait never ends CANCELLED.
                if result == _api.Result.CANCELLED and (self._stop.is_set() or self._closing):
                    break
                if result != _api.Result.OK:
                    self._failed(result)
                    break
                task = _PublicationTask(self, native_task)
                native_task = None
                if self._disarmed:
                    task.cancel()
                else:
                    try:
                        self._dispatch(task)
                    except BaseException:
                        # The dispatcher may have queued the callable and then
                        # thrown. Disarm that same callable before abandoning.
                        task.cancel()
                        raise
                task = None
        except BaseException as error:
            self._failed(_api.Result.RESOURCE_EXHAUSTED if isinstance(error, MemoryError)
                         else _api.Result.INTERNAL_ERROR)
            _api._log_failure("Live USD host dispatch failed")
        finally:
            self._done.set()

    @property
    def creations(self):
        """Retained creation owners, including those whose caller dropped them."""
        with self._activity:
            return tuple(self._creations)

    def prepare_creation(self, seed, live_uri):
        """Freeze a seed on this exact host without issuing a service request."""
        return self._prepare_creation(seed, live_uri, None)

    def _prepare_creation(self, seed, live_uri, on_retained):
        """Internal host workflow seam; retain association before native work."""
        if not isinstance(live_uri, str):
            raise TypeError("live_uri must be a string")
        if threading.get_ident() != self._creator:
            raise _api.LiveError(_api.Result.BUSY)
        # Bind both wrappers before reserving native creation state. An interrupt
        # anywhere after preparation starts leaves an accessible closeable owner.
        owner = Creation(self)
        owner._native = _api._query(self._native.allocate_creation())
        owner._admitting = True
        try:
            with self._activity:
                if self._closing or self._disarmed or self._closed:
                    raise _api.LiveError(_api.Result.NOT_INITIALIZED)
                if len(self._creations) >= self._max_creations:
                    raise _api.LiveError(_api.Result.RESOURCE_EXHAUSTED)
                self._creations.add(owner)
            # Do not infer native admission from Python delivery. Any exception
            # retains a wrapper that can be inspected and explicitly closed.
            if on_retained is not None:
                on_retained(owner)
            result = self._native.prepare_creation(owner._native, seed, live_uri)
        finally:
            owner._admitting = False
        if result != _api.Result.OK:
            owner.close()
            raise _api.LiveError(result)
        return owner

    def prepare_retained_creation(self, seed, live_uri):
        """Freeze without RPC; process retention survives interrupted delivery."""
        if not isinstance(live_uri, str):
            raise TypeError("live_uri must be a string")
        if threading.get_ident() != self._creator:
            raise _api.LiveError(_api.Result.BUSY)
        if self._closing or self._disarmed or self._closed:
            raise _api.LiveError(_api.Result.NOT_INITIALIZED)
        return _api._query(self._native.prepare_retained_creation(seed, live_uri))

    def acquire_retained_creation(self, subject_id):
        """Transfer on the original host; preserve the workflow's own callbacks.

        A replacement context can own the Python wrapper, but does not restore
        an abandoned original dispatcher. Inspect/export/close such work there.
        Interrupted delivery leaves the owner in this context's creations.
        """
        subject_id = _subject(subject_id)
        if threading.get_ident() != self._creator:
            raise _api.LiveError(_api.Result.BUSY)
        owner = Creation(self)
        owner._native = _api._query(self._native.allocate_creation())
        owner._admitting = True
        try:
            with self._activity:
                if self._closing or self._disarmed or self._closed:
                    raise _api.LiveError(_api.Result.NOT_INITIALIZED)
                if len(self._creations) >= self._max_creations:
                    raise _api.LiveError(_api.Result.RESOURCE_EXHAUSTED)
                self._creations.add(owner)
            result = self._native.acquire_retained_creation(owner._native, subject_id)
        finally:
            owner._admitting = False
        if result != _api.Result.OK:
            owner.close()  # Failed transfer has no native owner to discard.
            raise _api.LiveError(result)
        return owner

    def close(self, timeout_ms=30000):
        deadline = time.monotonic() + _api._timeout(timeout_ms) / 1000
        worker = (getattr(_api._local, "runtime", None) is not None or
                  getattr(_api._local, "host_task", None) is not None)
        if not self._close_lock.acquire(timeout=0 if worker else max(0, deadline - time.monotonic())):
            return _api.Result.BUSY if worker else _api.Result.TIMEOUT
        try:
            if self._retiring:
                return _api.Result.BUSY
            if self._closed:
                return _api.Result.OK
            self._closing = True
            with self._activity:
                if any(owner._admitting for owner in self._creations):
                    return _api.Result.BUSY
            result = _api.Result(self._native.close())
            if result != _api.Result.OK:
                return result  # Keep the consumer alive to retire existing work.
            self._stop.set()
            self._native.interrupt_wait()
            if worker:
                return _api.Result.BUSY
            if not self._done.wait(_api._remaining(deadline) / 1000):
                return _api.Result.TIMEOUT
            self._thread.join()
            if not self._wait_running(_api._remaining(deadline) / 1000):
                return _api.Result.TIMEOUT
            self._retiring = True
        finally:
            self._close_lock.release()
        completed = False
        try:
            # User captures may run finalizers that reenter close. Retire them
            # outside locks, with success withheld until all cleanup finishes.
            self._dispatch = self._on_error = None
            with self._runtime._condition:
                self._runtime._hosts.discard(self)
            completed = True
        finally:
            with self._close_lock:
                self._closed = completed
                self._retiring = False
        return _api.Result.OK

    def _abandon(self):
        with self._activity:
            self._disarmed = True
        self._stop.set()
        if self._native is not None:
            self._native.abandon()
            self._native.interrupt_wait()

    def _wait_running(self, timeout=None):
        with self._activity:
            return self._activity.wait_for(lambda: not self._running, timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        result = self.close()
        if result != _api.Result.OK and exc_type is None:
            raise _api.LiveError(result)


class _PublicationTask:
    def __init__(self, context, native):
        self._context = context
        self._native = native

    def cancel(self):
        self._native = None

    def __call__(self):
        context = self._context
        previous = getattr(_api._local, "host_task", None)
        previous_runtime = getattr(_api._local, "runtime", None)
        with context._activity:
            if self._native is None or context._disarmed:
                self.cancel()
                return _api.Result.CANCELLED
            native = self._native
            context._running += 1
        try:
            _api._local.host_task = context
            _api._local.runtime = context._runtime
            result = _api.Result(native.run())
            if result == _api.Result.OK:
                self._native = None
            if result != _api.Result.OK:
                raise _api.LiveError(result)
            return result
        finally:
            try:
                native = None
            finally:
                try:
                    _api._local.host_task = previous
                finally:
                    try:
                        _api._local.runtime = previous_runtime
                    finally:
                        with context._activity:
                            context._running -= 1
                            context._activity.notify_all()


class CreationPhase(IntEnum):
    PREPARED = 0
    CREATING = 1
    CONFIRMED = 2
    UNISSUED_FAILURE = 3
    REJECTED = 4
    OUTCOME_UNKNOWN = 5


@dataclass(frozen=True)
class CreationOutcome:
    phase: CreationPhase
    result: _api.Result
    error: _api.RequestError
    subject_id: int
    started: bool
    closing: bool
    quiescent: bool


class Creation:
    """Owned one-shot layer creation, retained by context until explicit close.

    Start authorizes replacement at live_uri. Wait and outcome are thread-neutral;
    all other operations run on the original host. Wait OK means quiescence,
    so inspect outcome before preparing/activating the host layer. Close retires
    the native owner even if copied layer handles remain.
    Dropping a Python reference never cancels, retries or discards this workflow.
    """
    def __init__(self, context):
        self._context = context
        self._native = None
        self._admitting = False

    def start(self):
        return _api.Result(self._native.start())

    def wait(self, timeout_ms=30000):
        return _api.Result(self._native.wait(_api._timeout(timeout_ms)))

    @property
    def outcome(self):
        value = _api._query(self._native.outcome())
        return CreationOutcome(CreationPhase(value["phase"]), _api.Result(value["result"]),
                               _api._record(_api.RequestError, value["error"]),
                               value["subject_id"], value["started"], value["closing"], value["quiescent"])

    def wait_closed(self, timeout_ms=30000):
        """After close begins, wait off-host for retirement; close on-host again.

        Original-host calls only poll. Timeout preserves closing ownership and
        never releases USD objects. Before close returns INVALID_ARGUMENT.
        """
        if self._admitting:
            return _api.Result.BUSY
        timeout = _api._timeout(timeout_ms)
        if (getattr(_api._local, "runtime", None) is not None or
                getattr(_api._local, "host_task", None) is not None):
            timeout = 0  # A Python callback may own the task being retired.
        return _api.Result(self._native.wait_closed(timeout))

    def prepare_layer(self, timeout_ms=30000):
        return _api.Result(self._native.prepare_layer(_api._timeout(timeout_ms)))

    def activate(self):
        return _api.Result(self._native.activate())

    @property
    def layer(self):
        return _api._query(self._native.layer())

    def close(self):
        """Explicit original-host retirement. Export any needed content with `Sdf.Layer.Export` first."""
        if self._admitting:
            return _api.Result.BUSY
        result = _api.Result(self._native.close())
        if result == _api.Result.OK:
            with self._context._activity:
                self._context._creations.discard(self)
        return result


@dataclass(frozen=True)
class RetainedCreationInfo:
    subject_id: int
    live_uri: str


def _subject(value, *, allow_zero=False):
    if type(value) is not int or not (0 if allow_zero else 1) <= value <= 0xffffffffffffffff:
        raise ValueError("subject must be an unsigned 64-bit integer" + ("" if allow_zero else " greater than zero"))
    return value


def next_retained_creation(after_subject=0):
    """Observe the next process-retained subject, or None; no context required.

    This is not a pin or an atomic scan. Acquisition can subsequently return
    NOT_FOUND/BUSY. Leased and retiring native owners remain visible.
    """
    result, value = _api._live.next_retained_creation(_subject(after_subject, allow_zero=True))
    if result == _api.Result.NOT_FOUND:
        return None
    return RetainedCreationInfo(**_api._query((result, value)))


def release_retained_creation(subject_id):
    """Original-host explicit cleanup. Export any needed content with `Sdf.Layer.Export` first.

    No configured Python runtime/context is required. BUSY keeps the native
    owner discoverable; allow its outstanding jobs to retire before retrying.
    """
    return _api.Result(_api._live.release_retained_creation(_subject(subject_id)))


def prepare_retained_creation(seed, live_uri, *, context):
    return context.prepare_retained_creation(seed, live_uri)


def acquire_retained_creation(subject_id, *, context):
    return context.acquire_retained_creation(subject_id)


def set_publication_context(dispatch, *, max_pending=64, on_error=None):
    if not callable(dispatch) or (on_error is not None and not callable(on_error)):
        raise TypeError("dispatch and optional on_error must be callable")
    if type(max_pending) is not int or not 1 <= max_pending <= 256:
        raise ValueError("max_pending must be an integer in [1, 256]")
    runtime = _api._current()
    context = None
    started = False
    try:
        with runtime._condition:
            if runtime._closing or runtime._closed or _api._stopping:
                raise _api.LiveError(_api.Result.NOT_INITIALIZED)
            if len(runtime._hosts) >= runtime._max_host_contexts:
                raise _api.LiveError(_api.Result.RESOURCE_EXHAUSTED)
            context = PublicationContext(runtime, dispatch, on_error, max_pending)
            runtime._hosts.add(context)
            context._thread.start()
            started = True
            context._native = _api._query(_api._live.create_publication_context(max_pending))
            context._admitted.set()
            return context
    except BaseException:
        if context is not None:
            context._abandon()
            context._admitted.set()
            if started:
                context._thread.join()
            with runtime._condition:
                runtime._hosts.discard(context)
        raise


def prepare_creation(seed, live_uri, *, context):
    return context.prepare_creation(seed, live_uri)


def save_layer(layer, timeout_ms=30000):
    return _api.Result(_api._live.save_layer(layer, _api._timeout(timeout_ms)))
