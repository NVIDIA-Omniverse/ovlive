# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-thread session catalogue; service identity never comes from storage files."""
import asyncio
from dataclasses import dataclass
import sys
import threading

_api = sys.modules[__package__]


@dataclass(frozen=True)
class SessionRow:
    session_id: str
    uri: str
    root_uri: str
    name: str
    created_by: str | None = None
    created_time_ms: int | None = None


class SessionOperation:
    """Retained request/result. Cancelling a waiter does not cancel a mutation."""
    def __init__(self, catalog, kind, argument):
        self._catalog, self._kind, self._argument = catalog, kind, argument
        self.request = None
        self._root = None
        self._outcome = None
        self._delivery = None
        self._delivery_lock = threading.Lock()
        self._provisional = False
        self.projection_result = _api.Result.OK
        self._future = catalog._loop.create_future()

    @property
    def kind(self):
        return self._kind

    @property
    def argument(self):
        return self._argument

    @property
    def outcome(self):
        self._catalog._on_host()
        if self._outcome is None:
            raise _api.LiveError(_api.Result.BUSY)
        return self._outcome

    async def wait(self):
        self._catalog._on_host()
        if self._outcome is not None:
            return self._outcome
        return await asyncio.shield(self._future)


class SessionCatalog:
    """One endpoint/base-layer scope with at most one issued request.

    The caller supplies a stable unsigned base-layer key and the identity used
    by the service. Polling is active only while requested by a UI/workflow.
    Mutations are never retried. Keep and acknowledge each mutation's receipt;
    unknown outcomes require an explicit preservation/reconciliation decision.
    This class neither opens layers nor owns stage/merge/storage policy.
    """
    def __init__(self, endpoint, base_layer_key, *, current_user, max_sessions=1024,
                 max_operations=64, poll_interval=2.0, loop=None, on_change=None):
        if on_change is not None and not callable(on_change):
            raise TypeError("on_change must be callable")
        if (not isinstance(base_layer_key, str) or not base_layer_key or
                any(c in base_layer_key for c in ("?", "#", "\x00"))):
            raise ValueError("base_layer_key must be a stable unsigned identity without query or fragment")
        if not isinstance(current_user, str):
            raise TypeError("current_user must be the service identity, or an empty string if unknown")
        for value in (max_sessions, max_operations):
            if type(value) is not int or not 1 <= value <= 4096:
                raise ValueError("catalogue limits must be integers in [1, 4096]")
        if type(poll_interval) not in (int, float) or not 0.1 <= poll_interval <= 3600:
            raise ValueError("poll_interval must be between 0.1 and 3600 seconds")
        self._thread = threading.get_ident()
        self._loop = loop if loop is not None else asyncio.get_running_loop()
        self.endpoint, self.base_layer_key = endpoint, base_layer_key
        self.current_user = current_user
        self._max_sessions, self._max_operations = max_sessions, max_operations
        self._interval = poll_interval
        self._rows, self._confirmed = {}, {}
        self._operations = []
        self._pending = None
        self._timer = None
        self._active = self._closing = self._closed = False
        self.metadata_result = _api.Result.NOT_INITIALIZED
        self.poll_result = _api.Result.OK
        self.view_result = _api.Result.OK
        self._on_change = on_change
        self._notifying = False

    def refresh_projection(self):
        """Project current state synchronously without issuing another RPC.

        on_change(catalog) is a host-thread, read-only observer returning Result.
        A failed observer cannot erase a service receipt or keep merge enabled.
        """
        self._on_host()
        if self._notifying:
            raise _api.LiveError(_api.Result.BUSY)
        if self._on_change is None:
            return self.view_result
        self._notifying = True
        self.view_result = _api.Result.OK
        try:
            self.view_result = _api.Result(self._on_change(self))
        except BaseException as error:
            self.view_result = error.result if isinstance(error, _api.LiveError) else (
                _api.Result.RESOURCE_EXHAUSTED if isinstance(error, MemoryError) else _api.Result.INTERNAL_ERROR)
        finally:
            self._notifying = False
        return self.view_result

    def _can_mutate(self):
        self._on_host()
        if self._notifying:
            raise _api.LiveError(_api.Result.BUSY)

    def _on_host(self):
        if threading.get_ident() != self._thread:
            raise _api.LiveError(_api.Result.BUSY)

    @property
    def rows(self):
        self._on_host()
        return tuple(self._rows.values())

    @property
    def operations(self):
        self._on_host()
        return tuple(self._operations)

    @property
    def closed(self):
        return self._closed

    def can_merge(self, session_id):
        self._on_host()
        row = self._rows.get(session_id)
        return bool(not self._closing and self.metadata_result == _api.Result.OK and
                    self.view_result == _api.Result.OK and
                    row is not None and row.created_by and row.created_by == self.current_user and
                    not any(op.kind == "delete" and op.argument == session_id and
                            (op._outcome is None or op._outcome[0] == _api.Result.OUTCOME_UNKNOWN)
                            for op in self._operations))

    def set_active(self, active):
        self._can_mutate()
        if type(active) is not bool:
            raise TypeError("active must be bool")
        if active and self._closing:
            raise _api.LiveError(_api.Result.NOT_INITIALIZED)
        self._active = active
        self._cancel_timer()
        if active and self._pending is None:
            self.refresh()

    def _cancel_timer(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule_poll(self):
        if self._active and not self._closing and self._timer is None and self._pending is None:
            try:
                self._timer = self._loop.call_later(self._interval, self._poll)
                self.poll_result = _api.Result.OK
            except RuntimeError:
                self.poll_result = _api.Result.INTERNAL_ERROR
                self._active = False

    def _poll(self):
        self._timer = None
        if self._active and not self._closing and self._pending is None:
            self.refresh()

    def refresh(self):
        self._can_mutate()
        if self._pending is not None and self._pending.kind == "list":
            return self._pending
        return self._start("list", None)

    def create(self, name):
        self._can_mutate()
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError("session name must be nonempty text without NUL")
        if len(self._rows) >= self._max_sessions:
            raise _api.LiveError(_api.Result.RESOURCE_EXHAUSTED)
        return self._start("create", name)

    def delete(self, session_id):
        self._can_mutate()
        if session_id not in self._rows:
            raise KeyError(session_id)
        return self._start("delete", session_id)

    def _start(self, kind, argument):
        self._can_mutate()
        if self._closing:
            raise _api.LiveError(_api.Result.NOT_INITIALIZED)
        if self._pending is not None:
            raise _api.LiveError(_api.Result.BUSY)
        if kind != "list" and len(self._operations) >= self._max_operations:
            raise _api.LiveError(_api.Result.RESOURCE_EXHAUSTED)
        operation = SessionOperation(self, kind, argument)
        def deliver(*values):
            # The public API may admit the request and invoke this callback
            # even if its caller is interrupted before receiving the Request.
            with operation._delivery_lock:
                operation._delivery = values if len(values) == 2 else (values[0], None)
            try:
                self._loop.call_soon_threadsafe(self._complete, operation)
            except RuntimeError:
                # Loop teardown cannot erase a mutation. reconcile()/close()
                # can recover its result directly from the retained Request.
                pass
        # Allocate callback/receipt storage before publishing in-flight state;
        # failure here has not issued an RPC and must not strand close BUSY.
        if kind != "list":
            self._operations.append(operation)
        self._pending = operation
        self._cancel_timer()
        if self.refresh_projection() != _api.Result.OK:
            self._finish(operation, self.view_result, None)
            return operation  # Projection rejected before request admission.
        try:
            if kind == "list":
                operation.request = _api.list_sessions_with_callback(self.endpoint, self.base_layer_key, deliver)
            elif kind == "create":
                operation.request = _api.create_session_with_callback(self.endpoint, self.base_layer_key, argument, deliver)
            else:
                # URI-only public calls use the latest global Python runtime.
                # This host must keep deletion with the endpoint's actual owner.
                operation.request = self.endpoint._runtime._submit(
                    "delete_session", (self._rows[argument].uri,), deliver, False)
        except _api.LiveError as error:
            self._finish(operation, error.result, None)
        except BaseException:
            # Keep admission uncertainty provisional: a later callback can
            # still supply the original confirmed result without another RPC.
            self._finish(operation, _api.Result.INTERNAL_ERROR if kind == "list" else _api.Result.OUTCOME_UNKNOWN,
                         None, provisional=kind != "list")
            self.reconcile()
            raise
        return operation

    def _complete(self, operation):
        self._on_host()
        if operation._outcome is not None and not operation._provisional:
            return
        with operation._delivery_lock:
            outcome = operation._delivery
        if outcome is None:
            if operation.request is None:
                return
            outcome = operation.request.outcome
        result, payload = outcome if isinstance(outcome, tuple) else (outcome, None)
        self._finish(operation, result, payload)

    def _finish(self, operation, result, payload, *, provisional=False):
        # Preserve the service receipt before any fallible UI projection. An
        # inability to build rows must not turn a confirmed Create into a retry.
        operation._outcome = (result, payload)
        operation._provisional = provisional
        if not provisional and self._pending is operation:
            self._pending = None
        try:
            if operation.kind == "list" and result == _api.Result.OK:
                # Build and validate the whole snapshot before replacing UI state.
                listed = {item.session_id: SessionRow(item.session_id, item.uri,
                    _api.make_layer_uri(item.uri, "root.live"), item.name,
                    item.created_by, item.created_time_ms) for item in payload}
                rows = dict(self._confirmed)
                rows.update(listed)
                if len(listed) != len(payload):
                    operation.projection_result = _api.Result.INTERNAL_ERROR
                elif len(rows) > self._max_sessions:
                    operation.projection_result = _api.Result.RESOURCE_EXHAUSTED
                else:
                    self._rows = rows
            elif result == _api.Result.OK and operation.kind == "create":
                row = SessionRow(payload.session_id, payload.uri,
                                 _api.make_layer_uri(payload.uri, "root.live"), operation.argument)
                confirmed, rows = dict(self._confirmed), dict(self._rows)
                confirmed[row.session_id] = rows[row.session_id] = row
                self._confirmed, self._rows = confirmed, rows
            elif result == _api.Result.OK and operation.kind == "delete":
                confirmed, rows = dict(self._confirmed), dict(self._rows)
                confirmed.pop(operation.argument, None)
                rows.pop(operation.argument, None)
                self._confirmed, self._rows = confirmed, rows
        except BaseException:
            operation.projection_result = _api.Result.INTERNAL_ERROR
        if operation.kind == "list":
            self.metadata_result = result if result != _api.Result.OK else operation.projection_result
        elif operation.projection_result != _api.Result.OK:
            self.metadata_result = operation.projection_result
        self.refresh_projection()
        if not operation._future.done():
            try:
                operation._future.set_result(operation._outcome)
            except RuntimeError:
                pass  # A closed loop cannot remove the retained outcome.
        self._schedule_poll()

    def reconcile(self):
        """Recover completion even if the host loop could not dispatch it."""
        self._can_mutate()
        operation = self._pending
        if operation is not None:
            with operation._delivery_lock:
                delivered = operation._delivery is not None
            if delivered or (operation.request is not None and operation.request.wait(0) == _api.Result.OK):
                self._complete(operation)

    def acknowledge(self, operation):
        """Release a consumed, known mutation result; never imply a retry is safe."""
        self._can_mutate()
        self.reconcile()
        if operation not in self._operations:
            raise ValueError("operation is not retained by this catalogue")
        if operation._outcome is None or operation._outcome[0] == _api.Result.OUTCOME_UNKNOWN:
            return _api.Result.BUSY
        if operation.request is not None:
            result = operation.request.close(0)
            if result != _api.Result.OK:
                return result
        self._operations.remove(operation)
        return _api.Result.OK

    def close(self):
        """Poll cleanup. Pending/unknown/unconsumed mutations retain ownership."""
        self._can_mutate()
        self._closing = True
        self.set_active(False)
        self.reconcile()
        self.refresh_projection()
        if self._pending is not None or self._operations:
            return _api.Result.BUSY
        self._closed = True
        return _api.Result.OK
