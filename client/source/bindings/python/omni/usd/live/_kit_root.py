# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained, one-shot root creation for one confirmed Kit session receipt."""
import asyncio
import sys

_api = sys.modules[__package__]


def _failure(error):
    return error.result if isinstance(error, _api.LiveError) else (
        _api.Result.RESOURCE_EXHAUSTED if isinstance(error, MemoryError) else _api.Result.INTERNAL_ERROR)


class SessionRoot:
    """Host-owned creation; neither close nor activation acknowledges its session.

    A successful service session is not proof that root.live exists. This owner
    freezes one seed, permits one CreateLayer attempt and retains its native
    outcome owner. Activation publishes a live layer, not a Kit stage
    attachment. Export any needed content with `Sdf.Layer.Export` before
    explicitly closing this owner.
    """
    def __init__(self, host, catalog, operation, root_uri):
        self.host, self.catalog, self.operation = host, catalog, operation
        self.root_uri = root_uri
        self.creation = None
        self.preparation_result = _api.Result.NOT_INITIALIZED
        self.result = _api.Result.NOT_INITIALIZED
        self.closed = False
        self.activated = False
        self._active = False
        self._start_attempted = False
        self._wait_task = None
        self.wait_result = _api.Result.NOT_INITIALIZED

    def _on_host(self):
        self.host._on_host()
        if self._active:
            raise _api.LiveError(_api.Result.BUSY)

    def _retain(self, creation):
        # Called after context retention and before native preparation. No USD,
        # callbacks, service work or allocating container update at this boundary.
        self.creation = creation

    def _prepare(self, seed):
        self._active = True
        try:
            self.host.publication_context._prepare_creation(seed, self.root_uri, self._retain)
            self.preparation_result = self.result = _api.Result.OK
        except BaseException as error:
            self.preparation_result = self.result = _failure(error)
            raise
        finally:
            self._active = False

    def _result(self, value):
        self.result = value
        return value

    def start(self):
        """Issue at most once, even if native result delivery is interrupted."""
        self._on_host()
        if self.closed or self.host._closing:
            return self._result(_api.Result.NOT_INITIALIZED)
        if self._start_attempted:
            return self._result(_api.Result.ALREADY_EXISTS)
        if self.preparation_result != _api.Result.OK:
            return self._result(self.preparation_result)
        self._start_attempted = True
        self._active = True
        try:
            self.result = self.creation.start()
            return self.result
        except BaseException as error:
            self.result = _failure(error)
            raise
        finally:
            self._active = False

    async def wait(self, timeout_ms=30000):
        """Wait off-host; cancelling a UI waiter leaves the native wait owned.

        OK means request quiescence, not creation success: inspect
        creation.outcome. A later explicit wait may continue after TIMEOUT;
        concurrent waiters share the first wait's deadline.
        """
        self._on_host()
        timeout = _api._timeout(timeout_ms)
        if self.closed or self.creation is None:
            return _api.Result.NOT_INITIALIZED
        if self._wait_task is None or self._wait_task.done():
            coroutine = self._wait_creation(timeout)
            try:
                self._wait_task = asyncio.create_task(coroutine)
            except BaseException:
                coroutine.close()
                raise
        return await asyncio.shield(self._wait_task)

    async def _wait_creation(self, timeout):
        try:
            self.wait_result = await asyncio.to_thread(self.creation.wait, timeout)
        except asyncio.CancelledError:
            # Only loop shutdown or an explicit task cancellation can reach the
            # owned task; ordinary UI cancellation is shielded. Native ownership
            # survives even when an executor wait outlives this event loop.
            self.wait_result = _api.Result.CANCELLED
            raise
        except BaseException as error:
            self.wait_result = _failure(error)
        return self.wait_result

    def activate(self, timeout_ms=30000):
        """Prepare and activate on the original host; never issue another Create.

        This bounded host operation may wait for bootstrap reads. The host must
        have valid credentials and a progressing service/transport. It does not
        attach a stage or authorize unguarded USD access.
        """
        self._on_host()
        timeout = _api._timeout(timeout_ms)
        if self.closed or self.host._closing or self.creation is None:
            return self._result(_api.Result.NOT_INITIALIZED)
        if self._wait_task is not None and not self._wait_task.done():
            return self._result(_api.Result.BUSY)
        self._active = True
        try:
            outcome = self.creation.outcome
            if not outcome.quiescent:
                return self._result(_api.Result.BUSY)
            if outcome.phase != _api.CreationPhase.CONFIRMED:
                return self._result(outcome.result if outcome.result != _api.Result.OK else _api.Result.BUSY)
            self.result = self.creation.prepare_layer(timeout)
            if self.result == _api.Result.OK:
                self.result = self.creation.activate()
                if self.result == _api.Result.OK:
                    self.activated = True
            return self.result
        except BaseException as error:
            self.result = _failure(error)
            raise
        finally:
            self._active = False

    @property
    def layer(self):
        self._on_host()
        if self.closed or not self.activated:
            raise _api.LiveError(_api.Result.NOT_INITIALIZED)
        return self.creation.layer

    def close(self):
        """Explicit retirement after any needed `Sdf.Layer.Export` and detachment; never auto-discard."""
        self._on_host()
        if self.closed:
            return _api.Result.OK
        if self._wait_task is not None and not self._wait_task.done():
            return _api.Result.BUSY
        self._active = True
        try:
            result = self.creation.close() if self.creation is not None else _api.Result.OK
            if result == _api.Result.OK:
                self.closed = True
                self.activated = False
            return result
        finally:
            self._active = False
