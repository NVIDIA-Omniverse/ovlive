# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kit lifecycle ownership; no Kit imports, frame pump, or authentication policy."""
from dataclasses import dataclass
import sys
import threading
import time

_api = sys.modules[__package__]


@dataclass
class EndpointRegistration:
    """Retains partial setup as well as successful auth/status subscriptions."""
    endpoint: object = None
    auth: object = None
    status: object = None


class KitHost:
    """One Kit-main-thread owner. Retain this object until close returns OK.

    submit_coroutine is called from the publication consumer thread: it must be
    thread-safe, queue the coroutine onto this host's event loop and return a
    Future (the Kit extension passes asyncio.run_coroutine_threadsafe bound to
    Kit's loop); tests drive explicit scheduling events.
    Callbacks supplied at endpoint registration run on bounded API consumers and
    must not touch USD directly. Session/UI policy belongs to the calling host.
    """
    def __init__(self, submit_coroutine, *, max_endpoints=64, max_catalogs=64, max_session_roots=64):
        if not callable(submit_coroutine):
            raise TypeError("submit_coroutine must be callable")
        if type(max_endpoints) is not int or not 1 <= max_endpoints <= 256:
            raise ValueError("max_endpoints must be an integer in [1, 256]")
        if type(max_catalogs) is not int or not 1 <= max_catalogs <= 256:
            raise ValueError("max_catalogs must be an integer in [1, 256]")
        if type(max_session_roots) is not int or not 1 <= max_session_roots <= 256:
            raise ValueError("max_session_roots must be an integer in [1, 256]")
        self._session_roots = []
        self._max_session_roots = max_session_roots
        self._submit = submit_coroutine
        self._thread = threading.get_ident()
        self.runtime = None
        self.publication_context = None
        self._endpoints = []
        self._max_endpoints = max_endpoints
        self._catalogs = []
        self._max_catalogs = max_catalogs
        self._closing = False
        self._closed = False

    def _on_host(self):
        if threading.get_ident() != self._thread:
            raise _api.LiveError(_api.Result.BUSY)

    def start(self, *, max_pending=64, **runtime_options):
        self._on_host()
        if self.runtime is not None or self._closing:
            raise _api.LiveError(_api.Result.BUSY)
        # The publication factory uses the latest Python owner. Keep
        # registration together so another owner cannot redirect this context.
        with _api._owners_lock:
            self.runtime = _api.initialize(**runtime_options)
            self.publication_context = _api.set_publication_context(self._dispatch, max_pending=max_pending)

    def _dispatch(self, job):
        async def apply():
            job()

        coroutine = apply()
        try:
            future = self._submit(coroutine)
            future.add_done_callback(lambda finished: self._retire(finished, job))
        except BaseException:
            coroutine.close()
            # PublicationContext also cancels this exact job if submission
            # queued it and then threw. Its failed state preserves local work.
            raise

    def _retire(self, future, job):
        result = _api.Result.OK
        try:
            if future.cancelled():
                result = _api.Result.CANCELLED
            else:
                error = future.exception()
                if error is not None:
                    result = error.result if isinstance(error, _api.LiveError) else _api.Result.INTERNAL_ERROR
        except BaseException:
            result = _api.Result.INTERNAL_ERROR
        if result != _api.Result.OK:
            job.cancel()
            # A cancelled scheduled coroutine cannot silently lose a native
            # publication while leaving the context apparently healthy.
            self.publication_context._failed(result)

    @property
    def endpoints(self):
        return tuple(self._endpoints)

    def register_endpoint(self, authority, *, on_auth_required, on_status, **options):
        self._on_host()
        if self.runtime is None or self._closing or self.publication_context is None:
            raise _api.LiveError(_api.Result.NOT_INITIALIZED)
        if not callable(on_auth_required) or not callable(on_status):
            raise TypeError("auth and status handlers must be callable")
        if len(self._endpoints) >= self._max_endpoints:
            raise _api.LiveError(_api.Result.RESOURCE_EXHAUSTED)
        registration = EndpointRegistration()
        self._endpoints.append(registration)
        try:
            registration.endpoint = self.runtime.register_endpoint(authority, **options)
        except BaseException:
            self._endpoints.remove(registration)
            raise
        registration.auth = _api.register_auth_required_callback(registration.endpoint, on_auth_required)
        registration.status = _api.register_status_callback(registration.endpoint, on_status)
        return registration

    @property
    def session_catalogs(self):
        return tuple(self._catalogs)

    def observe_status(self, callback, *, on_error=None):
        """Observe this owner's runtime even if another Python owner is newer."""
        self._on_host()
        if self._closing or self.runtime is None:
            raise _api.LiveError(_api.Result.NOT_INITIALIZED)
        return self.runtime._subscribe("status", None, callback, on_error)

    def session_catalog(self, registration, base_layer_key, **options):
        """Create a bounded catalogue on the running Kit host event loop."""
        self._on_host()
        if self._closing or self.runtime is None:
            raise _api.LiveError(_api.Result.NOT_INITIALIZED)
        if not any(item is registration for item in self._endpoints) or registration.endpoint is None:
            raise ValueError("endpoint registration does not belong to this host")
        self._catalogs[:] = [item for item in self._catalogs if not item.closed]
        if len(self._catalogs) >= self._max_catalogs:
            raise _api.LiveError(_api.Result.RESOURCE_EXHAUSTED)
        from ._kit_sessions import SessionCatalog
        catalog = SessionCatalog(registration.endpoint, base_layer_key, **options)
        self._catalogs.append(catalog)
        return catalog

    @property
    def session_roots(self):
        self._on_host()
        return tuple(self._session_roots)

    def prepare_session_root(self, catalog, operation, seed):
        """Freeze one root seed for a retained, confirmed CreateSession receipt.

        Repeated calls return the same owner and never reprepare/reissue. If
        preparation raises, find its owner in session_roots; both the original
        session receipt and any allocated native creation remain discoverable.
        """
        self._on_host()
        catalog._can_mutate()
        if self._closing or self.publication_context is None:
            raise _api.LiveError(_api.Result.NOT_INITIALIZED)
        if not any(item is catalog for item in self._catalogs):
            raise ValueError("catalogue does not belong to this host")
        if not any(item is operation for item in catalog.operations) or operation.kind != "create":
            raise ValueError("a retained CreateSession receipt is required")
        if operation._root is not None:
            return operation._root
        # A Python interruption after append but before receipt association
        # must not allocate another workflow for this same mutation.
        for retained in self._session_roots:
            if retained.operation is operation:
                operation._root = retained
                return retained
        result, created = operation.outcome
        if result != _api.Result.OK:
            raise _api.LiveError(result)
        root_uri = _api.make_layer_uri(created.uri, "root.live")
        self._session_roots[:] = [item for item in self._session_roots if not item.closed]
        if len(self._session_roots) >= self._max_session_roots:
            raise _api.LiveError(_api.Result.RESOURCE_EXHAUSTED)
        from ._kit_root import SessionRoot
        root = SessionRoot(self, catalog, operation, root_uri)
        self._session_roots.append(root)
        operation._root = root
        root._prepare(seed)
        return root

    def close(self, timeout_ms=0):
        """Stop opens/authoring; preserve auth and callbacks until USD retires.

        This does not save, discard edits, close a Kit stage, or authorize forced
        exit. Detach layers only after drain or any needed `Sdf.Layer.Export`.
        BUSY/TIMEOUT retain all owners for explicit later completion.
        """
        self._on_host()
        deadline = time.monotonic() + _api._timeout(timeout_ms) / 1000
        if self._closed:
            return _api.Result.OK
        self._closing = True
        catalog_result = _api.Result.OK
        for catalog in self._catalogs:
            result = catalog.close()
            if result != _api.Result.OK:
                catalog_result = result
        if self.publication_context is not None:
            result = self.publication_context.close(_api._remaining(deadline))
            if result != _api.Result.OK:
                return result
        if catalog_result != _api.Result.OK:
            return catalog_result
        if any(not root.closed for root in self._session_roots):
            return _api.Result.BUSY
        if self.runtime is not None:
            result = self.runtime.close(_api._remaining(deadline))
            if result != _api.Result.OK:
                return result
        self._endpoints.clear()
        self._catalogs.clear()
        self._session_roots.clear()
        self._closed = True
        return _api.Result.OK
