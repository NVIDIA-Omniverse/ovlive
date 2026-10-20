# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Literal control frames and deterministic proxy-boundary checks, without sockets."""
import asyncio
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
import http2_proxy as h2


class Writer:
    def __init__(self):
        self.bytes = bytearray()
        self.drained = asyncio.Event()
        self.transport = SimpleNamespace(abort=lambda: None)

    def write(self, data):
        self.bytes.extend(data)

    async def drain(self):
        self.drained.set()


def reader(data):
    stream = asyncio.StreamReader()
    stream.feed_data(data)
    stream.feed_eof()
    return stream


class Http2ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_literal_control_frames_and_validation(self):
        self.assertEqual(h2.reset(3).encode(), bytes.fromhex('00000403000000000300000008'))
        self.assertEqual(h2.goaway().encode(), bytes.fromhex('0000080700000000007fffffff00000000'))
        for stream in (0, -1, 2**31):
            with self.assertRaises(ValueError):
                h2.reset(stream)
        for code in (-1, 2**32):
            with self.assertRaises(ValueError):
                h2.reset(1, code)
        frame = await h2.read_frame(reader(bytes.fromhex('00000403000000000300000008')))
        self.assertEqual((frame.kind, frame.flags, frame.stream, frame.payload), (3, 0, 3, bytes.fromhex('00000008')))

    async def test_fragmented_header_group_preserves_all_bytes(self):
        first = bytes.fromhex('000002010000000001abcd')
        last = bytes.fromhex('000001090400000001ef')
        data = bytes.fromhex('00000100000000000112')
        source = reader(first + last + data)
        self.assertEqual(b''.join(f.encode() for f in await h2.read_group(source)), first + last)
        self.assertEqual(b''.join(f.encode() for f in await h2.read_group(source)), data)
        self.assertEqual(await h2.read_group(source), [])

    async def test_truncation_interleaving_reserved_bit_and_bounds_fail(self):
        cases = [b'\0', bytes.fromhex('000004030000000003000000'),
                 bytes.fromhex('000000090400000001'), bytes.fromhex('000000010000000001'),
                 bytes.fromhex('000000010000000001000000060000000000'),
                 bytes.fromhex('000000010000000001000000090400000003'),
                 bytes.fromhex('000000018400000001')[:5] + bytes.fromhex('80000001')]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                await h2.read_group(reader(raw))
        with patch.object(h2, 'MAX_FRAME', 3), self.assertRaises(ValueError):
            await h2.read_frame(reader(bytes.fromhex('000004000000000001')))
        with patch.object(h2, 'MAX_GROUP', 17), self.assertRaises(ValueError):
            await h2.read_group(reader(bytes.fromhex('000000010000000001000000090400000001')))

    async def test_reset_is_observed_bidirectional_and_does_not_close_connection(self):
        proxy = h2.Http2Proxy('unused:1')
        client, server = Writer(), Writer()
        connection = h2.Connection(1, proxy.target, None, client)
        connection.server, connection.active = server, {1, 5}
        proxy.connections.append(connection)
        faults = await proxy.reset_active()
        expected = bytes.fromhex('0000040300000000010000000800000403000000000500000008')
        self.assertEqual(bytes(client.bytes), expected)
        self.assertEqual(bytes(server.bytes), expected)
        self.assertEqual([f['stream'] for f in faults], [1, 5])
        self.assertEqual(connection.active, set())
        self.assertFalse(connection.closed)
        with self.assertRaises(AssertionError):
            await proxy.reset_active()

    async def test_pump_tracks_half_closed_requests_and_terminal_responses(self):
        proxy = h2.Http2Proxy('unused:1')
        client, server = Writer(), Writer()
        connection = h2.Connection(1, proxy.target, None, client)
        connection.server = server
        request = bytes.fromhex('000000010400000001000000000100000001')
        await proxy.pump(connection, reader(request), server, False)
        self.assertEqual(connection.active, {1})
        self.assertEqual(bytes(server.bytes), request)
        response = bytes.fromhex('000000010400000001000000000000000001000000010500000001')
        await proxy.pump(connection, reader(response), client, True)
        self.assertEqual(connection.active, set())
        self.assertEqual(bytes(client.bytes), response)
        self.assertTrue(all(set(e) == {'connection','direction','kind','flags','stream','size'} for e in proxy.events))

    async def test_ingress_rejection_answers_from_the_static_table_and_refunds_dropped_data(self):
        proxy = h2.Http2Proxy('unused:1')
        client, server = Writer(), Writer()
        connection = h2.Connection(1, proxy.target, None, client)
        connection.server = server
        proxy.reject_new_streams('envoy')
        headers, data = h2.Frame(1, 4, 1, b'\x82'), h2.Frame(0, 1, 1, b'x' * 5)
        await proxy.pump(connection, reader(headers.encode() + data.encode()), server, False)
        # The header block still reaches the backend's HPACK decoder, then the stream is cancelled.
        self.assertEqual(bytes(server.bytes), headers.encode() + h2.reset(1, 8).encode())
        self.assertEqual(bytes(client.bytes), h2.Frame(1, 5, 1, h2.INGRESS_RESPONSES['envoy']).encode() +
                         h2.Frame(8, 0, 0, struct.pack('>I', 5)).encode())
        self.assertEqual(connection.active, set())
        self.assertEqual([(e['action'], e['stream']) for e in proxy.events if e.get('action') == 'ingress-reject'],
                         [('ingress-reject', 1)])
        # A backend header block on the rejected stream would desynchronize HPACK.
        await proxy.pump(connection, reader(h2.Frame(1, 5, 1, b'\x88').encode()), client, True)
        self.assertEqual(proxy.errors, ['ingress rejection dropped a backend header block'])
        # The next new stream is forwarded.
        proxy.errors.clear(); server.bytes.clear()
        await proxy.pump(connection, reader(h2.Frame(1, 4, 3, b'\x82').encode()), server, False)
        self.assertEqual(bytes(server.bytes), h2.Frame(1, 4, 3, b'\x82').encode())
        self.assertEqual(connection.active, {3})
        for kind, count in (('other', 1), ('envoy', 0)):
            with self.assertRaises(ValueError):
                proxy.reject_new_streams(kind, count)
        self.assertEqual(h2.INGRESS_RESPONSES['http401'][:5], b'\x08\x03401')  # :status 401, no indexing.

    async def test_later_frames_on_a_rejected_stream_are_dropped_and_their_data_refunded(self):
        proxy = h2.Http2Proxy('unused:1')
        client, server = Writer(), Writer()
        connection = h2.Connection(1, proxy.target, None, client)
        connection.server = server
        proxy.reject_new_streams('http401')
        await proxy.pump(connection, reader(h2.Frame(1, 4, 1, b'\x82').encode()), server, False)
        server.bytes.clear(); client.bytes.clear(); proxy.events.clear()
        # Client RST_STREAM, stream WINDOW_UPDATE and DATA in a later group never reach the backend.
        later = [h2.reset(1, 8), h2.Frame(8, 0, 1, struct.pack('>I', 7)), h2.Frame(0, 1, 1, b'x' * 4)]
        await proxy.pump(connection, reader(b''.join(frame.encode() for frame in later)), server, False)
        self.assertEqual(bytes(server.bytes), b'')
        self.assertEqual(bytes(client.bytes), h2.Frame(8, 0, 0, struct.pack('>I', 4)).encode())
        # Backend DATA on the rejected stream is dropped and refunded to the backend's connection window.
        await proxy.pump(connection, reader(h2.Frame(0, 0, 1, b'y' * 6).encode()), client, True)
        self.assertEqual(bytes(client.bytes), h2.Frame(8, 0, 0, struct.pack('>I', 4)).encode())
        self.assertEqual(bytes(server.bytes), h2.Frame(8, 0, 0, struct.pack('>I', 6)).encode())
        self.assertEqual([(e['action'], e['direction'], e['kind'], e['stream']) for e in proxy.events],
                         [('ingress-drop', 'server', 3, 1), ('ingress-drop', 'server', 8, 1),
                          ('ingress-drop', 'server', 0, 1), ('ingress-drop', 'client', 0, 1)])
        self.assertEqual(proxy.errors, [])

    async def test_ingress_rejection_forwards_the_whole_header_block_before_the_reset(self):
        proxy = h2.Http2Proxy('unused:1')
        client, server = Writer(), Writer()
        connection = h2.Connection(1, proxy.target, None, client)
        connection.server = server
        proxy.reject_new_streams('envoy')
        headers, continuation = h2.Frame(1, 0, 1, b'\x82'), h2.Frame(9, 4, 1, b'\x86')
        await proxy.pump(connection, reader(headers.encode() + continuation.encode()), server, False)
        self.assertEqual(bytes(server.bytes), headers.encode() + continuation.encode() + h2.reset(1, 8).encode())
        self.assertEqual(bytes(client.bytes), h2.Frame(1, 5, 1, h2.INGRESS_RESPONSES['envoy']).encode())
        self.assertEqual([e['action'] for e in proxy.events], ['ingress-reject'])

    async def test_ingress_rejection_counts_new_streams_and_disarms(self):
        proxy = h2.Http2Proxy('unused:1')
        client, server = Writer(), Writer()
        connection = h2.Connection(1, proxy.target, None, client)
        connection.server = server
        proxy.reject_new_streams('http401', 2)
        streams = [h2.Frame(1, 4, stream, b'\x82') for stream in (1, 3, 5)]
        await proxy.pump(connection, reader(b''.join(f.encode() for f in streams)), server, False)
        self.assertEqual(bytes(server.bytes), streams[0].encode() + h2.reset(1, 8).encode() +
                         streams[1].encode() + h2.reset(3, 8).encode() + streams[2].encode())
        self.assertEqual(bytes(client.bytes), b''.join(h2.Frame(1, 5, stream, h2.INGRESS_RESPONSES['http401']).encode()
                                                       for stream in (1, 3)))
        self.assertEqual((proxy.ingress, proxy.rejected, connection.active),
                         ([], {(1, 1): 'http401', (1, 3): 'http401'}, {5}))
        proxy.reject_new_streams('envoy', 2)
        proxy.disarm()
        server.bytes.clear()
        await proxy.pump(connection, reader(h2.Frame(1, 4, 7, b'\x82').encode()), server, False)
        self.assertEqual(bytes(server.bytes), h2.Frame(1, 4, 7, b'\x82').encode())
        self.assertEqual((proxy.ingress, connection.active), ([], {5, 7}))

    async def test_a_client_that_closes_before_its_preface_is_teardown(self):
        proxy = h2.Http2Proxy('unused:1')
        connection = h2.Connection(1, proxy.target, reader(b''), Writer())
        await proxy.forward(connection)  # A TLS client that rejected the certificate.
        self.assertEqual((proxy.errors, connection.closed), ([], True))
        with self.assertRaisesRegex(ValueError, 'truncated HTTP/2 client preface'):
            await proxy.forward(h2.Connection(2, proxy.target, reader(h2.PREFACE[:5]), Writer()))

    async def test_goaway_fence_requires_exact_ping_ack_and_does_not_forward_it(self):
        proxy = h2.Http2Proxy('A:1')
        client, server = Writer(), Writer()
        connection = h2.Connection(1, proxy.target, None, client)
        connection.server, connection.active = server, {1}
        proxy.connections.append(connection)
        premature = h2.Frame(6, 1, 0, connection.drain_ping).encode()
        await proxy.pump(connection, reader(premature), server, False)
        self.assertFalse(connection.drain_ack.is_set())
        task = asyncio.create_task(proxy.drain_to('B:2'))
        try:
            await asyncio.wait_for(client.drained.wait(), 1)
            self.assertFalse(task.done())
            wrong = h2.Frame(6, 1, 0, b'wrongack').encode()
            await proxy.pump(connection, reader(wrong), server, False)
            self.assertFalse(connection.drain_ack.is_set())
            invalid_stream = h2.Frame(6, 1, 1, connection.drain_ping).encode()
            await proxy.pump(connection, reader(invalid_stream), server, False)
            self.assertFalse(connection.drain_ack.is_set())
            right = h2.Frame(6, 1, 0, connection.drain_ping).encode()
            await proxy.pump(connection, reader(right), server, False)
            self.assertEqual(await task, (connection,))
            self.assertEqual(bytes(server.bytes), premature + wrong + invalid_stream)
            self.assertEqual(proxy.target, 'B:2')
            self.assertEqual(connection.target, 'A:1')
            self.assertEqual(connection.active, {1})
            self.assertFalse(connection.closed)
            with self.assertRaises(AssertionError):
                await proxy.drain_to('C:3')
            with self.assertRaises(AssertionError):
                await proxy.reset_active(connections=[])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_error_evidence_is_bounded(self):
        proxy = h2.Http2Proxy('unused:1')
        with patch.object(h2, 'MAX_ERRORS', 2):
            for i in range(5):
                proxy.error(str(i))
        self.assertEqual(proxy.snapshot()['errors'], ['0', '1'])
        self.assertEqual(proxy.snapshot()['errors_dropped'], 3)

    async def test_response_byte_boundary_waits_for_forwarding_and_reset_wakes_observers(self):
        class ObservedEvent(asyncio.Event):
            def __init__(self):
                super().__init__()
                self.entered = asyncio.Event()

            async def wait(self):
                self.entered.set()
                return await super().wait()

        class HeldWriter(Writer):
            def __init__(self):
                super().__init__()
                self.release = asyncio.Event()

            async def drain(self):
                self.drained.set()
                await self.release.wait()

        proxy = h2.Http2Proxy('unused:1')
        client, server = HeldWriter(), Writer()
        connection = h2.Connection(1, proxy.target, None, client)
        connection.server = server
        proxy.connections.append(connection)
        await proxy.pump(connection, reader(bytes.fromhex('000000010400000001')), server, False)
        sent = asyncio.create_task(proxy.pump(connection, reader(bytes.fromhex('00000700000000000101020304050607')), client, True))
        observers = [asyncio.create_task(proxy.wait_response_bytes(connection, 1, n)) for n in (3, 7)]
        try:
            await asyncio.wait_for(client.drained.wait(), 1)
            self.assertEqual(connection.response_bytes, {1: 0})
            self.assertTrue(all(not task.done() for task in observers))
            client.release.set()
            await asyncio.wait_for(sent, 1)
            self.assertEqual(await asyncio.wait_for(asyncio.gather(*observers), 1), [7, 7])
            self.assertEqual(await proxy.wait_response_bytes(connection, 1, 7), 7)
            proxy.changed = ObservedEvent()
            pending = asyncio.create_task(proxy.wait_response_bytes(connection, 1, 8))
            observers.append(pending)
            await asyncio.wait_for(proxy.changed.entered.wait(), 1)
            await proxy.reset_active()
            with self.assertRaises(AssertionError):
                await asyncio.wait_for(pending, 1)
            self.assertEqual(connection.response_bytes, {})
        finally:
            client.release.set()
            for task in [sent, *observers]:
                task.cancel()
            await asyncio.gather(sent, *observers, return_exceptions=True)

    async def test_response_observation_rejects_padded_data_and_invalid_targets(self):
        proxy = h2.Http2Proxy('unused:1')
        client, server = Writer(), Writer()
        connection = h2.Connection(1, proxy.target, None, client)
        connection.server = server
        proxy.connections.append(connection)
        await proxy.pump(connection, reader(bytes.fromhex('000000010400000001')), server, False)
        with self.assertRaises(ValueError):
            await proxy.pump(connection, reader(bytes.fromhex('00000100080000000100')), client, True)
        for stream, minimum in ((0, 1), (1, 0), (1, True), (1, 2**64)):
            with self.assertRaises(ValueError):
                await proxy.wait_response_bytes(connection, stream, minimum)
        proxy.error('synthetic failure')
        with self.assertRaises(AssertionError):
            await proxy.wait_response_bytes(connection, 1, 1)
