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
"""Wire fixtures and a protocol-only OODF decoder; no server state logic."""

import base64
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import zlib

CONTENT_HASH_BLOCK_BYTES = 1 << 20

ERROR_INFO_TYPE = "type.googleapis.com/google.rpc.ErrorInfo"
RETRY_INFO_TYPE = "type.googleapis.com/google.rpc.RetryInfo"


def _fields(message):
    """Yield (field number, value) for each field of a protobuf-encoded message.

    Varints as ints, length-delimited fields as bytes. Enough to read the
    google.rpc.Status a refusal carries without the googleapis protos.
    """
    pos = 0

    def varint():
        nonlocal pos
        result = shift = 0
        while True:
            byte = message[pos]
            pos += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7

    while pos < len(message):
        key = varint()
        number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value = varint()
        elif wire_type == 2:
            length = varint()
            value = message[pos : pos + length]
            pos += length
        elif wire_type in (1, 5):
            width = 8 if wire_type == 1 else 4
            value = message[pos : pos + width]
            pos += width
        else:
            raise AssertionError(f"Unsupported protobuf wire type {wire_type}")
        yield number, value


def status_details(error):
    """The details a gRPC error carries in grpc-status-details-bin, by type URL."""
    raw = dict(error.trailing_metadata() or ()).get("grpc-status-details-bin")
    details = {}
    if raw is None:
        return details
    for number, value in _fields(raw):
        if number == 3:  # google.rpc.Status.details: repeated google.protobuf.Any
            any_fields = dict(_fields(value))
            details[any_fields.get(1, b"").decode()] = any_fields.get(2, b"")
    return details


def error_info(error):
    """(reason, domain, metadata) of the google.rpc.ErrorInfo a gRPC error carries, or None."""
    value = status_details(error).get(ERROR_INFO_TYPE)
    if value is None:
        return None
    reason, domain, metadata = "", "", {}
    for number, field in _fields(value):
        if number == 1:
            reason = field.decode()
        elif number == 2:
            domain = field.decode()
        elif number == 3:
            entry = dict(_fields(field))
            metadata[entry.get(1, b"").decode()] = entry.get(2, b"").decode()
    return reason, domain, metadata


def pack_ompf(delta):
    """An OMPF container holding `delta` as its only part."""
    header = 8 + 8
    return b"OMPF" + struct.pack("<IQ", 1, header) + delta


class _FlatBuffer:
    """Just enough of a FlatBuffers writer for the oversized deltas below.

    Laid out front to back: the root offset and the file identifier, then each
    table after its vtable, then what the table points at, so every unsigned
    offset points forward as the format requires. Little-endian, every scalar
    aligned to its size, as the verifier checks.
    """

    def __init__(self):
        self.data = bytearray()

    def align(self, size, remainder=0):
        while len(self.data) % size != remainder:
            self.data.append(0)

    def put(self, raw):
        position = len(self.data)
        self.data += raw
        return position

    def offset_to(self, at, target):
        """Point the unsigned offset at `at` to `target`, which lies after it."""
        assert target > at, "FlatBuffers offsets point forward"
        struct.pack_into("<I", self.data, at, target - at)

    def table(self, slots, inline_size):
        """A vtable naming each field's inline slot (0 = absent), then its
        table, which starts with its offset back to the vtable. Returns the
        table's position; the caller writes the inline fields after it."""
        self.align(4)
        vtable = self.put(struct.pack(f"<HH{len(slots)}H", 4 + 2 * len(slots), inline_size, *slots))
        self.align(4)
        table = self.put(struct.pack("<i", 0))
        struct.pack_into("<i", self.data, table, table - vtable)
        return table


# Union discriminants of OmniObjectDelta's StructCommand (schema order).
_COMMAND_DELETE_SECTION = 2
_COMMAND_REORDER_CHILDREN = 4


def delta_of_entries(struct_commands, set_fields):
    """An OODF delta of `struct_commands` DeleteSection commands and `set_fields`
    SetFields — the entries one delta carries, counted by the server across its
    struct commands, set fields and time samples.

    Every element of a kind points at one shared table with every field at its
    default: a valid FlatBuffer whose vector lengths are the counts, a few
    hundred KB rather than the tens of MB a client's encoder would write. A
    server refuses it for its counts before it walks any element.
    """
    buf = _FlatBuffer()
    buf.put(struct.pack("<I", 0) + b"OODF")
    # Delta fields: isDiff, baseVersion, structCommands_type, structCommands, setFields.
    delta = buf.table([0, 0, 4, 8, 12], 16)
    buf.put(struct.pack("<III", 0, 0, 0))
    buf.offset_to(0, delta)

    buf.align(4)
    types = buf.put(struct.pack("<I", struct_commands))
    buf.put(bytes([_COMMAND_DELETE_SECTION]) * struct_commands)
    buf.align(4)
    commands = buf.put(struct.pack("<I", struct_commands) + b"\0" * 4 * struct_commands)
    buf.align(4)
    fields = buf.put(struct.pack("<I", set_fields) + b"\0" * 4 * set_fields)

    shared = buf.table([], 4)
    for vector, count in ((commands, struct_commands), (fields, set_fields)):
        for i in range(count):
            buf.offset_to(vector + 4 + 4 * i, shared)
    buf.offset_to(delta + 4, types)
    buf.offset_to(delta + 8, commands)
    buf.offset_to(delta + 12, fields)
    return bytes(buf.data)


def delta_of_children_list(children):
    """An OODF delta of one ReorderChildren command whose children list names
    `children` ids (1..children) under section 0."""
    buf = _FlatBuffer()
    buf.put(struct.pack("<I", 0) + b"OODF")
    # Delta fields: isDiff, baseVersion, structCommands_type, structCommands.
    delta = buf.table([0, 0, 4, 8], 12)
    buf.put(struct.pack("<II", 0, 0))
    buf.offset_to(0, delta)

    buf.align(4)
    types = buf.put(struct.pack("<IB", 1, _COMMAND_REORDER_CHILDREN))
    buf.align(4)
    commands = buf.put(struct.pack("<II", 1, 0))
    # ReorderChildren fields: sectionId, childrenListId, childrenList.
    reorder = buf.table([0, 0, 4], 8)
    buf.put(struct.pack("<I", 0))
    buf.align(8, 4)  # the u64 elements after the length word are 8-aligned
    children_list = buf.put(struct.pack("<I", children))
    buf.put(struct.pack(f"<{children}Q", *range(1, children + 1)))

    buf.offset_to(delta + 4, types)
    buf.offset_to(delta + 8, commands)
    buf.offset_to(commands + 4, reorder)
    buf.offset_to(reorder + 4, children_list)
    return bytes(buf.data)


def content_hash(value):
    """Hex content hash an external value's valueOrExtHash must carry.

    SHA-256 over the concatenated SHA-256 digests of the value's consecutive
    1 MiB blocks; an empty value hashes to SHA-256 of the empty string. This is
    the clients' definition, not plain SHA-256 of the bytes, written out here
    so the suite stays independent of the service code.
    """
    outer = hashlib.sha256()
    for start in range(0, len(value), CONTENT_HASH_BLOCK_BYTES):
        outer.update(hashlib.sha256(value[start : start + CONTENT_HASH_BLOCK_BYTES]).digest())
    return outer.hexdigest()


def fixture(name):
    data = json.loads((Path(__file__).parent / "fixtures" / "synthetic.json").read_text())
    return zlib.decompress(base64.b64decode(data[name], validate=True))


def capture():
    return json.loads((Path(__file__).parent / "fixtures" / "captured.json").read_text())


def capture_golden():
    return json.loads((Path(__file__).parent / "fixtures" / "captured-golden.json").read_text())[
        "normalized_signature"
    ]


def unpack_ompf(payload):
    if len(payload) < 16 or payload[:4] != b"OMPF":
        raise AssertionError("Response is not a complete OMPF container")
    count = struct.unpack_from("<I", payload, 4)[0]
    if not 1 <= count <= (len(payload) - 8) // 8:
        raise AssertionError("Invalid OMPF part count")
    offsets = list(struct.unpack_from(f"<{count}Q", payload, 8)) + [len(payload)]
    if offsets[0] < 8 + count * 8 or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise AssertionError("Invalid OMPF part offsets")
    return [payload[a:b] for a, b in zip(offsets, offsets[1:])]


def decode(settings, payload):
    parts = unpack_ompf(payload)
    result = subprocess.run(
        [settings.decoder, "--json", "--validate-externals"],
        input=base64.b64encode(payload),
        capture_output=True,
        timeout=settings.timeout,
    )
    if result.returncode:
        raise AssertionError(f"OODF decoder failed (exit {result.returncode})")
    signature = json.loads(result.stdout)
    blobs = {content_hash(p): p for p in parts[1:]}
    for value in signature["setFields"] + signature["timeSamples"]:
        if value["extValueSize"]:
            blob = blobs.get(value["valueOrExtHash"])
            assert blob is not None and len(blob) == value["extValueSize"], (
                "Missing/corrupt external value"
            )
    return signature


def field_order(signature):
    return [
        (v["sectionId"], v["keyName"])
        for v in sorted(signature["setFields"], key=lambda v: v["setOrder"])
    ]


def snapshot(settings, payload, base_version=0):
    signature = decode(settings, payload)
    assert signature["isDiff"] is True and signature["baseVersion"] == base_version, (
        "Invalid catch-up envelope"
    )
    orders = [value["setOrder"] for value in signature["setFields"]]
    # Absolute counters may differ across implementations; clients require
    # globally unique slots and relative authoring order, not dense numbering.
    assert len(set(orders)) == len(orders), "Catch-up field ordering slots collide"
    return signature


def apply_snapshot(signature, baseline=None):
    """Fold a corrected catch-up diff onto a client baseline, not onto server state.

    Catch-up commands contain disjoint upserts and retractions. Raw inbound
    command streams are intentionally unsupported: their order is significant.
    """
    sections, fields, samples = (
        ({}, {}, {}) if baseline is None else tuple(dict(x) for x in baseline)
    )
    for command in signature["structCommands"]:
        sid = command["sectionId"]
        if command["type"] == "CommandDiffDeleteSection":
            sections.pop(sid, None)
            fields = {key: val for key, val in fields.items() if key[0] != sid}
            samples = {key: val for key, val in samples.items() if key[0] != sid}
        elif command["type"] == "CommandDiffChangeSection":
            sections[sid] = (
                command["parentId"],
                command["sectionName"],
                command["sectionType"],
                command["sectionOrder"],
            )
        else:
            raise AssertionError(f"Unexpected catch-up command: {command['type']}")
    for value in signature["setFields"]:
        fields[(value["sectionId"], value["keyName"])] = (
            value["valueOrExtHash"],
            value["extValueSize"],
        )
    for value in signature["timeSamples"]:
        samples[(value["sectionId"], value["time"])] = (
            value["valueOrExtHash"],
            value["extValueSize"],
        )
    return sections, fields, samples
