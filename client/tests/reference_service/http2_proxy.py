# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded HTTP/2 fault proxy for local reference-service tests.

Opaque frames stay byte-for-byte intact, including HPACK state; an ingress
rejection answers from the static table and forwards the rejected header block
before resetting it. A rejected client-streaming request, such as UpdateLayer,
therefore starts its backend handler, which counts an attempt and is then
cancelled; an authenticating ingress such as Envoy ext_authz forwards nothing.
A rejected unary request never starts its handler, because its message never
arrives. Injection occurs between complete header blocks; no headers or
payload content enters evidence.
RFC 9113 sections 4.3, 6.4 and 6.8 define the boundaries/control frames:
https://www.rfc-editor.org/rfc/rfc9113.html
It terminates TLS when given a server context and forwards plaintext HTTP/2 to
the backend; it is not an HPACK decoder or a general HTTP/2 implementation.
"""
import asyncio
from dataclasses import dataclass
import struct

PREFACE = b'PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n'
MAX_FRAME = 1024 * 1024
MAX_GROUP = 2 * MAX_FRAME
MAX_CONNECTIONS = 64
MAX_STREAMS = 256
MAX_EVENTS = 4096
MAX_ERRORS = 64


@dataclass(frozen=True)
class Frame:
    kind: int
    flags: int
    stream: int
    payload: bytes

    def encode(self):
        if not 0 <= self.kind <= 255 or not 0 <= self.flags <= 255 or not 0 <= self.stream < 2**31:
            raise ValueError('invalid frame metadata')
        if len(self.payload) > MAX_FRAME:
            raise ValueError('frame exceeds proxy bound')
        return len(self.payload).to_bytes(3, 'big') + bytes((self.kind, self.flags)) + struct.pack('>I', self.stream) + self.payload


def reset(stream, code=8):
    if not 0 < stream < 2**31 or not 0 <= code < 2**32:
        raise ValueError('invalid reset target/code')
    return Frame(3, 0, stream, struct.pack('>I', code))


def goaway():
    # Initial graceful-drain announcement accepts every potentially issued RPC;
    # it never fabricates evidence permitting replay of a mutation.
    return Frame(7, 0, 0, struct.pack('>II', 2**31 - 1, 0))


def _literal(name, value):
    """HPACK literal without indexing, new name, no Huffman (RFC 7541 section 6.2.2)."""
    name, value = name.encode(), value.encode()
    if len(name) > 126 or len(value) > 126:
        raise ValueError('ingress header exceeds the fixture profile')
    return b'\x00' + bytes([len(name)]) + name + bytes([len(value)]) + value


# Responses built from the static table and literals without indexing, so an
# injected answer changes neither HPACK dynamic table.
INGRESS_RESPONSES = {
    # A front that is not gRPC-aware: HTTP 401 with a bearer challenge.
    'http401': b'\x08\x03401' + _literal('content-type', 'text/plain') + _literal('www-authenticate', 'Bearer'),
    # An Envoy ext_authz denial of a gRPC request: trailers-only, grpc-status 16.
    'envoy': b'\x88' + _literal('content-type', 'application/grpc') + _literal('grpc-status', '16'),
}


async def read_frame(reader):
    try:
        header = await reader.readexactly(9)
    except asyncio.IncompleteReadError as error:
        if not error.partial:
            return None
        raise ValueError('truncated HTTP/2 frame header') from error
    length = int.from_bytes(header[:3], 'big')
    if length > MAX_FRAME:
        raise ValueError('frame exceeds proxy bound')
    try:
        payload = await reader.readexactly(length)
    except asyncio.IncompleteReadError as error:
        raise ValueError('truncated HTTP/2 frame payload') from error
    # Keep the reserved stream bit verbatim by rejecting it in this bounded
    # fixture profile rather than silently rewriting a forwarded frame.
    stream = int.from_bytes(header[5:], 'big')
    if stream >= 2**31:
        raise ValueError('reserved stream bit outside proxy profile')
    return Frame(header[3], header[4], stream, payload)


async def read_group(reader):
    first = await read_frame(reader)
    if first is None:
        return []
    if first.kind == 9:
        raise ValueError('orphan HTTP/2 continuation')
    group, size = [first], 9 + len(first.payload)
    if first.kind in (1, 5) and not first.flags & 4:
        while True:
            part = await read_frame(reader)
            if part is None or part.kind != 9 or part.stream != first.stream:
                raise ValueError('interleaved or truncated HTTP/2 header block')
            size += 9 + len(part.payload)
            if size > MAX_GROUP:
                raise ValueError('header block exceeds proxy bound')
            group.append(part)
            if part.flags & 4:
                break
    return group


class Connection:
    def __init__(self, identity, target, reader, writer):
        self.id, self.target, self.reader, self.client = identity, target, reader, writer
        self.server = None
        self.to_client, self.to_server = asyncio.Lock(), asyncio.Lock()
        self.active = set()
        self.response_bytes = {}
        self.highest = 0
        self.draining = False
        self.closed = False
        self.drain_ping = b'drain' + identity.to_bytes(3, 'big')
        self.drain_ack = asyncio.Event()


class Http2Proxy:
    def __init__(self, target, *, ssl_context=None):
        self.target, self.ssl_context, self.connections, self.tasks = target, ssl_context, [], set()
        self.events, self.errors = [], []
        self.errors_dropped = 0
        self.ingress, self.rejected = [], {}
        self.closing = False
        self.changed = asyncio.Event()

    def error(self, message):
        if len(self.errors) < MAX_ERRORS:
            self.errors.append(message)
        else:
            self.errors_dropped += 1
        self.changed.set()

    def record(self, **event):
        if len(self.events) >= MAX_EVENTS:
            raise ValueError('HTTP/2 evidence capacity exceeded')
        self.events.append(event)

    def reject_new_streams(self, kind, count=1):
        """Answers the next `count` new client streams at the proxy, as an authenticating ingress does."""
        if kind not in INGRESS_RESPONSES or type(count) is not int or count < 1:
            raise ValueError('unknown ingress rejection')
        self.ingress.extend([kind] * count)

    def disarm(self):
        self.ingress.clear()

    async def start(self):
        # A client stalled in its TLS handshake is not yet a connection close()
        # can abort; this failure-only bound replaces asyncio's 60 s default,
        # so close() waits at most 5 s for such a client.
        self.listener = await asyncio.start_server(
            self.accept, '127.0.0.1', 0, ssl=self.ssl_context,
            ssl_handshake_timeout=5 if self.ssl_context else None)
        self.address = '127.0.0.1:' + str(self.listener.sockets[0].getsockname()[1])
        return self.address

    def accept(self, reader, writer):
        if self.closing or len(self.connections) >= MAX_CONNECTIONS:
            writer.transport.abort()
            if not self.closing:
                self.error('HTTP/2 connection capacity exceeded')
            return
        connection = Connection(len(self.connections) + 1, self.target, reader, writer)
        self.connections.append(connection)
        task = asyncio.create_task(self.forward(connection))
        self.tasks.add(task)
        def retired(done):
            self.tasks.discard(done)
            connection.closed = True
            connection.client.transport.abort()
            if connection.server:
                connection.server.transport.abort()
            if not done.cancelled() and done.exception():
                self.error(repr(done.exception()))
            self.changed.set()
        task.add_done_callback(retired)

    async def pump(self, connection, reader, writer, to_client):
        lock = connection.to_client if to_client else connection.to_server
        direction = 'client' if to_client else 'server'
        while group := await read_group(reader):
            # Streams this group opens and the proxy answers as the ingress, and
            # the connection window of the DATA it drops on rejected streams,
            # which returns to the sender: the client or the backend.
            injected, refund = [], 0
            async with lock:
                response_counts = []
                for frame in group:
                    rejected = (connection.id, frame.stream) in self.rejected
                    if not to_client and frame.kind == 1 and frame.stream > connection.highest and self.ingress:
                        if not frame.stream & 1:
                            raise ValueError('invalid or over-capacity client stream')
                        connection.highest = frame.stream
                        self.rejected[(connection.id, frame.stream)] = self.ingress.pop(0)
                        injected.append(frame.stream)
                        # The backend's HPACK decoder consumes the block; the reset cancels the stream.
                        writer.write(frame.encode())
                        continue
                    if not to_client and frame.kind == 9 and frame.stream in injected:
                        writer.write(frame.encode())
                        continue
                    if rejected:
                        if frame.kind == 0:
                            refund += len(frame.payload)
                        if to_client and frame.kind in (1, 9):
                            self.error('ingress rejection dropped a backend header block')
                        self.record(action='ingress-drop', connection=connection.id, direction=direction,
                                    kind=frame.kind, stream=frame.stream)
                        continue
                    if not to_client and frame.kind == 1 and frame.stream > connection.highest:
                        if not frame.stream & 1 or len(connection.active) >= MAX_STREAMS:
                            raise ValueError('invalid or over-capacity client stream')
                        connection.highest = frame.stream
                        connection.active.add(frame.stream)
                        connection.response_bytes[frame.stream] = 0
                    if to_client and frame.kind == 0 and frame.stream in connection.active:
                        if frame.flags & 8:
                            raise ValueError('padded DATA outside byte-observation profile')
                        response_counts.append((frame.stream, len(frame.payload)))
                    if frame.kind == 3 or to_client and frame.kind in (0, 1) and frame.flags & 1:
                        connection.active.discard(frame.stream)
                        connection.response_bytes.pop(frame.stream, None)
                    if (not to_client and connection.draining and frame.kind == 6 and frame.stream == 0 and
                            frame.flags == 1 and frame.payload == connection.drain_ping):
                        connection.drain_ack.set()
                        # This PING was originated by the proxy, not the backend.
                        continue
                    writer.write(frame.encode())
                    self.record(connection=connection.id, direction=direction,
                                kind=frame.kind, flags=frame.flags, stream=frame.stream, size=len(frame.payload))
                for stream in injected:
                    writer.write(reset(stream, 8).encode())
                await writer.drain()
                for stream, size in response_counts:
                    if stream in connection.active:
                        connection.response_bytes[stream] += size
            for stream in injected:
                kind = self.rejected[(connection.id, stream)]
                async with connection.to_client:
                    connection.client.write(Frame(1, 0x5, stream, INGRESS_RESPONSES[kind]).encode())
                    await connection.client.drain()
                self.record(action='ingress-reject', kind=kind, connection=connection.id, stream=stream)
            if refund:
                sender, sender_lock = ((connection.server, connection.to_server) if to_client
                                       else (connection.client, connection.to_client))
                async with sender_lock:
                    sender.write(Frame(8, 0, 0, struct.pack('>I', refund)).encode())
                    await sender.drain()
            self.changed.set()

    async def forward(self, connection):
        pumps = []
        try:
            try:
                preface = await connection.reader.readexactly(len(PREFACE))
            except asyncio.IncompleteReadError as error:
                if error.partial:
                    raise ValueError('truncated HTTP/2 client preface') from error
                return  # A client that closed first, such as one that rejected the TLS certificate.
            if preface != PREFACE:
                raise ValueError('proxy requires HTTP/2 prior knowledge')
            host, port = connection.target.rsplit(':', 1)
            upstream, connection.server = await asyncio.open_connection(host, int(port))
            connection.server.write(preface)
            await connection.server.drain()
            pumps = [asyncio.create_task(self.pump(connection, connection.reader, connection.server, False)),
                     asyncio.create_task(self.pump(connection, upstream, connection.client, True))]
            done, _ = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (ConnectionError, OSError):
            pass  # Peer socket teardown is ordinary; malformed frames are not.
        finally:
            for task in pumps:
                task.cancel()
            results = await asyncio.gather(*pumps, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception) and not isinstance(result, (ConnectionError, OSError)):
                    self.error(repr(result))
            connection.closed = True
            connection.client.transport.abort()
            if connection.server:
                connection.server.transport.abort()
            self.changed.set()

    async def wait_response_bytes(self, connection, stream, minimum):
        """Observe DATA forwarded on an existing response; no payload retention."""
        if (connection not in self.connections or type(stream) is not int or not 0 < stream < 2**31 or
                type(minimum) is not int or not 0 < minimum < 2**64):
            raise ValueError('invalid response observation')
        while True:
            # No await separates clearing this notification and checking state;
            # an observation cannot miss a transition on this event loop.
            self.changed.clear()
            if self.errors or connection.closed or stream not in connection.active:
                raise AssertionError('response ended before requested byte boundary')
            if connection.response_bytes.get(stream, 0) >= minimum:
                return connection.response_bytes[stream]
            await self.changed.wait()

    async def reset_active(self, *, connections=None, code=8):
        """Reset the observed streams in both directions; keep TCP alive.

        Select by observed connection/stream identity, not guessed RPC names.
        Service gates and client generation/cursor assertions prove which logical
        subscriptions the test interrupted. No HPACK inspection is required.
        """
        injected = []
        for connection in tuple(self.connections) if connections is None else connections:
            if connection.closed or connection.server is None:
                continue
            async with connection.to_server, connection.to_client:
                for stream in sorted(connection.active):
                    raw = reset(stream, code).encode()
                    connection.client.write(raw)
                    connection.server.write(raw)
                    record = dict(action='reset', connection=connection.id, stream=stream, code=code)
                    self.record(**record)
                    injected.append(record)
                connection.active.clear()
                connection.response_bytes.clear()
                await connection.client.drain()
                await connection.server.drain()
            self.changed.set()
        if not injected:
            raise AssertionError('HTTP/2 reset fault had no live stream')
        return injected

    async def drain_to(self, target):
        """Announce graceful drain; existing RPCs stay on their original backend."""
        previous = tuple(c for c in self.connections if not c.closed and not c.draining and c.server is not None)
        if not previous:
            raise AssertionError('HTTP/2 drain fault had no connection')
        self.target = target
        for connection in previous:
            async with connection.to_client:
                connection.client.write(goaway().encode())
                connection.client.write(Frame(6, 0, 0, connection.drain_ping).encode())
                connection.draining = True
                self.record(action='goaway', connection=connection.id, last_stream=2**31 - 1, code=0)
                await connection.client.drain()
            await asyncio.wait_for(connection.drain_ack.wait(), 5)
        return previous

    def snapshot(self):
        return dict(events=list(self.events), errors=list(self.errors), errors_dropped=self.errors_dropped,
                    connections=[dict(id=c.id, target=c.target, highest=c.highest, active=sorted(c.active),
                                      response_bytes=dict(c.response_bytes),
                                      draining=c.draining, closed=c.closed) for c in self.connections])

    async def close(self):
        self.closing = True
        self.listener.close()
        # Server.wait_closed waits for accepted connections. Aborting both
        # transports ends every forwarding task at a read or drain, so each
        # task runs its own teardown to completion before the listener closes.
        for connection in self.connections:
            connection.client.transport.abort()
            if connection.server:
                connection.server.transport.abort()
        await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
        await self.listener.wait_closed()
        if self.errors:
            raise AssertionError(f'HTTP/2 proxy errors: {self.errors}')
