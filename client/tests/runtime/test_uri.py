# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise actual compiled URI exports, with independent stdlib expectations.

These tests prove the URI component only, not a USD resolver or a shipped plugin.
"""
import ctypes as C
import ipaddress
import json
import os
from pathlib import Path
import random
import unittest
from urllib.parse import quote

OK, INVALID, UNSUPPORTED, EXHAUSTED, SMALL = 0, 1, 12, 13, 18
ABI = 0x10000


class Parts(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32),
                ("authority_offset", C.c_uint64), ("session_id_offset", C.c_uint64),
                ("layer_name_offset", C.c_uint64), ("canonical_uri_offset", C.c_uint64),
                ("is_layer", C.c_uint32)]


def encoded(value):
    return value.encode("utf-8") if isinstance(value, str) else value


class UriTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("LIVE_TEST_URI_LIBRARY")
        if not path or not Path(path).is_file():
            raise RuntimeError("LIVE_TEST_URI_LIBRARY must identify the built URI contract probe")
        cls.library = C.CDLL(path)
        cls.functions = {}
        for name in ("MakeSessionUri", "MakeLayerUri", "CombineWithBaseUri"):
            function = getattr(cls.library, "omniUsdLive" + name)
            function.argtypes = [C.c_char_p, C.c_char_p, C.c_void_p, C.POINTER(C.c_uint64)]
            function.restype = C.c_uint32
            cls.functions[name] = function
        cls.parse = cls.library.omniUsdLiveParseUri
        cls.parse.argtypes = [C.c_char_p, C.POINTER(Parts), C.c_void_p, C.POINTER(C.c_uint64)]
        cls.parse.restype = C.c_uint32

    def build(self, name, first, second):
        function = self.functions[name]
        size = C.c_uint64()
        self.assertEqual(function(encoded(first), encoded(second), None, C.byref(size)), OK)
        buffer = C.create_string_buffer(size.value)
        self.assertEqual(function(encoded(first), encoded(second), buffer, C.byref(size)), OK)
        self.assertEqual(buffer.raw[-1], 0)
        return buffer.value.decode("utf-8")

    def parse_uri(self, uri):
        parts = Parts(C.sizeof(Parts), ABI)
        size = C.c_uint64()
        self.assertEqual(self.parse(encoded(uri), C.byref(parts), None, C.byref(size)), OK)
        buffer = C.create_string_buffer(size.value)
        self.assertEqual(self.parse(encoded(uri), C.byref(parts), buffer, C.byref(size)), OK)
        strings = [buffer.raw[offset:].split(b"\0", 1)[0].decode("utf-8") for offset in
                   (parts.authority_offset, parts.session_id_offset, parts.layer_name_offset,
                    parts.canonical_uri_offset)]
        self.assertEqual(parts.is_layer, bool(strings[2]))
        return strings

    def test_authority_canonicalization_dns_ipv4_ipv6_ports(self):
        cases = {"Example.COM:443": "example.com", "LOCALHOST.": "localhost",
                 "127.0.0.1:0443": "127.0.0.1", "10.2.3.4:1": "10.2.3.4:1",
                 "[2001:0DB8:0000:0000:0000:0000:0000:0001]:443": "[2001:db8::1]",
                 "[::1]:65535": "[::1]:65535", "[::]": "[::]",
                 "[1:0:0:2:0:0:3:4]": "[1::2:0:0:3:4]",
                 "[::ffff:127.0.0.1]": "[::ffff:7f00:1]"}
        for authority, canonical in cases.items():
            with self.subTest(authority=authority):
                self.assertEqual(self.build("MakeSessionUri", authority, "id"), f"live://{canonical}/id")

    def test_opaque_sessions_percent_utf8_and_nested_layer_names(self):
        session = "opaque/session %2F?#🦉"
        layer = "folder ü/literal %2F?.live"
        uri = self.build("MakeSessionUri", "LIVE.example:443", session)
        self.assertEqual(uri, "live://live.example/" + quote(session, safe="-._~"))
        full = self.build("MakeLayerUri", uri, layer)
        self.assertEqual(full, uri + "/" + quote(layer, safe="/-._~"))
        self.assertEqual(self.parse_uri(full), ["live.example", session, layer, full])
        self.assertEqual(self.parse_uri(uri), ["live.example", session, "", uri])

    def test_equivalent_escapes_share_canonical_resource(self):
        self.assertEqual(self.parse_uri("live://LOCALHOST:443/%73ession/f%6flder/root%2elive"),
                         ["localhost", "session", "folder/root.live", "live://localhost/session/folder/root.live"])
        self.assertEqual(self.parse_uri("live://localhost/a%2fb/root.live")[1], "a/b")
        self.assertEqual(self.parse_uri("live://localhost/a%252fb/root.live")[1], "a%2fb")

    def test_invalid_uris_rejected_without_writing_outputs(self):
        invalid = ["", "https://host/session/root.live", "live://123/id/root.live",
                   "live://host", "live://host/", "live://host/id/", "live://host/id/root.usda",
                   "live://user@host/id/root.live", "live://host/id/root.live?token=x",
                   "live://host/id/root.live#fragment", "live://host/id/../root.live",
                   "live://host/id/%2e%2e/root.live", "live://host/id//root.live",
                   "live://host/id/a%2froot.live", "live://host/id/a%5croot.live",
                   "live://host/id%/root.live", "live://host/id%2/root.live", "live://host/id%GG/root.live",
                   "live://host/%00/root.live", "live://host/%0A/root.live", "live://host/%7f/root.live",
                   "live://host/%c2%80/root.live", "live://host/%c0%af/root.live",
                   "live://host/%ed%a0%80/root.live", "live://host/%f4%90%80%80/root.live",
                   "live://host/%ff/root.live", "live://host/%e2%82/root.live", "live://host/ü/root.live",
                   "live://host/./root.live", "live://host/../root.live", "live://host:0/id/root.live"]
        for uri in invalid:
            with self.subTest(uri=uri):
                parts = Parts(C.sizeof(Parts), ABI, 101, 102, 103, 104, 105)
                before = bytes(parts)
                buffer = C.create_string_buffer(b"z" * 256)
                size = C.c_uint64(len(buffer))
                self.assertEqual(self.parse(encoded(uri), C.byref(parts), buffer, C.byref(size)), INVALID)
                self.assertEqual(bytes(parts), before)
                self.assertEqual(buffer.raw, b"z" * 256 + b"\0")

    def test_invalid_authorities(self):
        invalid = ["", "1", "12345", "127.1", "0127.0.0.1", "127.00.0.1", "127.0.0.01", "256.0.0.1", "[::g]", "::1",
                   "[::1", "[::1]garbage", "[::1]:", "[fe80::1%25eth0]", "host:", "host:65536",
                   "host:443:80", "host:service", "-host", "host-", "ho_st", "host..name", "ü.example",
                   "user@host", "host/path", "host?token", "host#fragment", "host\\name", "host name",
                   "a" * 64 + ".com", "a." * 127 + "z"]
        for authority in invalid:
            with self.subTest(authority=authority):
                size = C.c_uint64()
                self.assertEqual(self.functions["MakeSessionUri"](encoded(authority), b"id", None, C.byref(size)), INVALID)

    def test_relative_layer_resolution_stays_in_session(self):
        self.assertEqual(self.build("CombineWithBaseUri", "live://host/s/a/root.live", "b/child.live"),
                         "live://host/s/a/b/child.live")
        self.assertEqual(self.build("CombineWithBaseUri", "live://host/s", "root.live"),
                         "live://host/s/root.live")
        for relative in ("/root.live", "../root.live", "a/../root.live", "./root.live", "", "file.usda"):
            size = C.c_uint64()
            self.assertEqual(self.functions["CombineWithBaseUri"](
                b"live://host/s/a/root.live", encoded(relative), None, C.byref(size)), INVALID)

    def test_size_query_small_buffer_null_arguments_and_abi_versions(self):
        for name, first, second in (("MakeSessionUri", b"host", b"id"),
                                    ("MakeLayerUri", b"live://host/id", b"root.live"),
                                    ("CombineWithBaseUri", b"live://host/id", b"root.live")):
            function = self.functions[name]
            size = C.c_uint64(1)
            buffer = C.create_string_buffer(b"sentinel")
            self.assertEqual(function(first, second, buffer, C.byref(size)), SMALL)
            self.assertGreater(size.value, 1)
            self.assertEqual(buffer.value, b"sentinel")
            self.assertEqual(function(None, second, None, C.byref(size)), INVALID)
            self.assertEqual(function(first, None, None, C.byref(size)), INVALID)
            self.assertEqual(function(first, second, None, None), INVALID)
        parts = Parts(C.sizeof(Parts), ABI)
        size = C.c_uint64(1)
        before = bytes(parts)
        buffer = C.create_string_buffer(b"sentinel")
        self.assertEqual(self.parse(b"live://host/id", C.byref(parts), buffer, C.byref(size)), SMALL)
        self.assertEqual(bytes(parts), before)
        self.assertEqual(buffer.value, b"sentinel")
        parts.abi_version = 0x20000
        self.assertEqual(self.parse(b"live://host/id", C.byref(parts), None, C.byref(size)), UNSUPPORTED)
        parts.struct_size = 4
        self.assertEqual(self.parse(b"live://host/id", C.byref(parts), None, C.byref(size)), INVALID)
        self.assertEqual(self.parse(b"live://host/id", None, None, C.byref(size)), INVALID)

    def test_extended_output_struct_tail_untouched(self):
        class Extended(C.Structure):
            _fields_ = [("parts", Parts), ("tail", C.c_uint64 * 8)]
        extended = Extended(Parts(C.sizeof(Extended), ABI + 1), (C.c_uint64 * 8)(*[0xDEADBEEF] * 8))
        size = C.c_uint64(256)
        buffer = C.create_string_buffer(256)
        self.assertEqual(self.parse(b"live://host/id/root.live", C.byref(extended.parts), buffer, C.byref(size)), OK)
        self.assertEqual(list(extended.tail), [0xDEADBEEF] * 8)
        self.assertEqual(extended.parts.struct_size, C.sizeof(Extended))

    def test_input_bounds_are_resource_errors(self):
        size = C.c_uint64()
        self.assertEqual(self.functions["MakeSessionUri"](b"host", b"x" * 65537, None, C.byref(size)), EXHAUSTED)
        self.assertEqual(self.functions["MakeSessionUri"](b"host", encoded("🦉" * 10000), None, C.byref(size)), EXHAUSTED)

    def test_seeded_independent_oracle_round_trips(self):
        seed = int(os.environ.get("LIVE_TEST_SEED", "42"))
        rng = random.Random(seed)
        alphabet = "abXYZ09-._~ /%?#:é中🦉"
        history = []
        artifact_dir = os.environ.get("LIVE_TEST_ARTIFACT_DIR")
        try:
            for index in range(1000):
                session = "s" + "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 35)))
                # Each layer component avoids slash ambiguity and dot traversal.
                layer = "/".join("n" + "".join(rng.choice(alphabet.replace("/", ""))
                                             for _ in range(rng.randrange(1, 20))) for _ in range(rng.randrange(1, 5))) + ".live"
                ipv6 = ipaddress.IPv6Address(rng.getrandbits(128))
                authority = f"[{ipv6.exploded.upper()}]:443"
                expected_authority = f"[{ipv6.compressed}]"
                expected = "live://" + expected_authority + "/" + quote(session, safe="-._~") + "/" + quote(layer, safe="/-._~")
                history.append({"index": index, "session": session, "layer": layer, "authority": authority})
                with self.subTest(seed=seed, index=index):
                    actual = self.build("MakeLayerUri", self.build("MakeSessionUri", authority, session), layer)
                    self.assertEqual(actual, expected)
                    self.assertEqual(self.parse_uri(actual), [expected_authority, session, layer, expected])
        finally:
            if artifact_dir:
                Path(artifact_dir, "uri-history.json").write_text(json.dumps(
                    {"schema_version": 1, "master_seed": seed, "operations": history}, ensure_ascii=True, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
