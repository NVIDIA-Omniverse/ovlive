# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Bounded public RPC helpers and resource ownership for the conformance suite."""

from contextlib import ExitStack
import queue
import threading
import time
import uuid

import grpc
from nvidia.omniverse.live.v1alpha import live_service_pb2 as pb
from nvidia.omniverse.live.v1alpha import live_service_pb2_grpc as rpc

# Put on a join's outgoing queue to send a JoinSession client keepalive
# instead of a message.
KEEPALIVE = object()


class Stream:
    def __init__(self, call, timeout):
        self.call, self.timeout = call, timeout
        self.events = queue.Queue(maxsize=4096)
        self.stopped = threading.Event()
        self.worker = threading.Thread(target=self._read, daemon=True)
        self.worker.start()

    def _read(self):
        try:
            for message in self.call:
                while not self.stopped.is_set():
                    try:
                        self.events.put(message, timeout=0.1)
                        break
                    except queue.Full:
                        pass
                if self.stopped.is_set():
                    return
            terminal = EOFError("Stream ended unexpectedly")
        except Exception as exc:
            terminal = exc
        while not self.stopped.is_set():
            try:
                self.events.put(terminal, timeout=0.1)
                return
            except queue.Full:
                pass

    def next(self, deadline=None):
        remaining = self.timeout if deadline is None else max(0, deadline - time.monotonic())
        try:
            item = self.events.get(timeout=remaining)
        except queue.Empty:
            raise AssertionError("Timed out waiting for a required stream message") from None
        if isinstance(item, Exception):
            raise item
        return item

    def delivery(self):
        deadline = time.monotonic() + self.timeout
        first = self.next(deadline)
        assert first.WhichOneof("response_type") == "header", "Expected header before chunks"
        header = first.header
        assert header.total_size <= 128 * 1024 * 1024, "Unbounded response size"
        data = bytearray()
        while len(data) < header.total_size:
            message = self.next(deadline)
            assert message.WhichOneof("response_type") == "chunk", "Header interrupted content"
            assert message.chunk.data, "Empty content chunk"
            data.extend(message.chunk.data)
            assert len(data) <= header.total_size, "Chunks exceed declared total_size"
        return header, bytes(data)

    def caught_up(self, object_id, sequence):
        header, data = self.delivery()
        assert not data and header.total_size == 0
        assert (header.object_id, header.sequence_num) == (object_id, sequence)
        assert header.client_timestamp_microseconds_since_epoch == 0
        assert header.server_timestamp_microseconds_since_epoch == 0

    def close(self):
        self.stopped.set()
        self.call.cancel()
        self.worker.join(timeout=2)
        assert not self.worker.is_alive(), "Stream reader failed to terminate after cancellation"


class Client:
    def __init__(self, settings, endpoint=None):
        self.settings = settings
        self.channel = settings.channel(endpoint)
        self.stub = rpc.LiveServiceStub(self.channel)
        self.resources = []
        self.streams = ExitStack()

    def call(self, name, request, *, metadata=None):
        response = getattr(self.stub, name)(
            request,
            metadata=self.settings.metadata() if metadata is None else metadata,
            timeout=self.settings.timeout,
        )
        # Even an unexpectedly accepted negative test must clean up its resource.
        if name == "CreateSession" and response.session_id:
            self.resources.append(response.session_id)
        return response

    def session(self, path=None, name=None):
        response = self.call(
            "CreateSession",
            pb.CreateSessionRequest(
                path=path or self.settings.base_path, name=name or f"conformance-{uuid.uuid4().hex}"
            ),
        )
        assert response.session_id
        return response.session_id

    def create(self, session, payload, layer="layer", chunk_size=65536, operation_id=""):
        requests = [
            pb.CreateLayerRequest(
                header=pb.CreateLayerHeader(
                    session_id=session, layer_name=layer, operation_id=operation_id
                )
            )
        ]
        requests.extend(
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=payload[i : i + chunk_size]))
            for i in range(0, len(payload), chunk_size)
        )
        return self.call("CreateLayer", iter(requests))

    def update(
        self,
        session,
        object_id,
        payload,
        layer="layer",
        client_id="writer",
        timestamp=123456,
        operation_id="",
    ):
        requests = [
            pb.UpdateLayerRequest(
                header=pb.UpdateLayerHeader(
                    session_id=session,
                    layer_name=layer,
                    object_id=object_id,
                    client_id=client_id,
                    timestamp_microseconds_since_epoch=timestamp,
                    operation_id=operation_id,
                )
            )
        ]
        requests.extend(
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=payload[i : i + 65536]))
            for i in range(0, len(payload), 65536)
        )
        return self.call("UpdateLayer", iter(requests))

    def chained_requests(
        self,
        session,
        object_id,
        payload,
        operation_id,
        previous_operation_id,
        layer="layer",
        client_id="writer",
        timestamp=123456,
        no_wait=False,
    ):
        """The messages of a chained UpdateLayer: the chained_header, then the content."""
        header = pb.UpdateLayerHeader(
            session_id=session,
            layer_name=layer,
            object_id=object_id,
            client_id=client_id,
            timestamp_microseconds_since_epoch=timestamp,
            operation_id=operation_id,
        )
        requests = [
            pb.UpdateLayerRequest(
                chained_header=pb.ChainedUpdateLayerHeader(
                    header=header,
                    previous_operation_id=previous_operation_id,
                    no_wait=no_wait,
                )
            )
        ]
        requests.extend(
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=payload[i : i + 65536]))
            for i in range(0, len(payload), 65536)
        )
        return requests

    def chained_update(
        self, session, object_id, payload, operation_id, previous_operation_id, **kw
    ):
        """A chained UpdateLayer, waited for (see chained_requests for the keywords)."""
        requests = self.chained_requests(
            session, object_id, payload, operation_id, previous_operation_id, **kw
        )
        return self.call("UpdateLayer", iter(requests))

    def chained_update_async(
        self, session, object_id, payload, operation_id, previous_operation_id, timeout=None, **kw
    ):
        """Start a chained UpdateLayer without waiting for it: its grpc future.

        `future.result()` waits for the answer; `future.cancel()` cancels the call.
        """
        requests = self.chained_requests(
            session, object_id, payload, operation_id, previous_operation_id, **kw
        )
        return self.stub.UpdateLayer.future(
            iter(requests),
            metadata=self.settings.metadata(),
            timeout=self.settings.timeout if timeout is None else timeout,
        )

    def unended_update_async(self, requests):
        """Start an UpdateLayer that sends `requests` and never ends its stream.

        Returns the call's future: cancel it to lose the call before its end of
        stream, as a broken connection would. The request stream is released
        when the client closes, after the call is cancelled, so it never sends
        an end of stream the server could commit on.
        """
        released = threading.Event()

        def stream():
            yield from requests
            released.wait()

        future = self.stub.UpdateLayer.future(
            stream(), metadata=self.settings.metadata(), timeout=self.settings.timeout
        )
        self.streams.callback(released.set)
        self.streams.callback(future.cancel)
        return future

    def subscribe(self, session, object_id=0, sequence=0, layer="layer", client_id="reader"):
        call = self.stub.SubscribeReadLayer(
            pb.SubscribeReadLayerRequest(
                session_id=session,
                layer_name=layer,
                object_id=object_id,
                sequence_num=sequence,
                client_id=client_id,
            ),
            metadata=self.settings.metadata(),
            timeout=110,
        )
        stream = Stream(call, self.settings.timeout)
        self.streams.callback(stream.close)
        return stream

    def join(self, session, keepalive=False):
        """Open a JoinSession; `keepalive` sets the header's `client_keepalive`.

        Returns the response stream and the outgoing queue: put bytes to send
        a message, or `KEEPALIVE` to send a keepalive. Nothing is sent unless
        the test puts it, so a test controls when the join is silent.
        """
        outgoing = queue.Queue()
        stopped = threading.Event()

        def requests():
            yield pb.JoinSessionRequest(
                header=pb.JoinSessionHeader(
                    session_id=session,
                    user_application="conformance",
                    client_keepalive=keepalive,
                )
            )
            while not stopped.is_set():
                try:
                    message = outgoing.get(timeout=0.1)
                except queue.Empty:
                    continue
                if message is KEEPALIVE:
                    yield pb.JoinSessionRequest(keepalive=pb.JoinSessionKeepalive())
                else:
                    yield pb.JoinSessionRequest(message=pb.SessionMessageRequest(data=message))

        stream = Stream(
            self.stub.JoinSession(requests(), metadata=self.settings.metadata(), timeout=110),
            self.settings.timeout,
        )
        self.streams.callback(stream.close)
        self.streams.callback(stopped.set)
        return stream, outgoing

    def close(self):
        failures = []
        try:
            self.streams.close()
        except Exception as exc:
            failures.append(str(exc))
        for session in reversed(self.resources):
            try:
                self.call("DeleteSession", pb.DeleteSessionRequest(session_id=session))
            except grpc.RpcError as exc:
                if exc.code() != grpc.StatusCode.NOT_FOUND:
                    failures.append(f"DeleteSession({session}): {exc.code().name}")
        self.channel.close()
        assert not failures, "Resource cleanup failed: " + "; ".join(failures)
