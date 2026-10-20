# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent, bounded OMPF oracle for root numeric fields, snapshots and aggregates.

This is deliberately not a general USD/OMPF codec. It uses only struct and the
published schema; no generated or production codec is imported. Unsupported
commands, samples, external values and types other than int32/finite double fail
closed. None denotes field deletion, not a numeric value. Every field record survives in Delta,
including repeated keys; dict-returning decode_update folds them in wire order.
"""
from dataclasses import dataclass
import math
import struct
from typing import Mapping

MAX_BYTES = 1024 * 1024
MAX_FIELDS = 4096


class WireError(ValueError):
    pass


@dataclass(frozen=True)
class Field:
    name: str
    value: int | float | None
    set_order: int


@dataclass(frozen=True)
class Delta:
    is_diff: bool
    base_version: int
    fields: tuple[Field, ...]


def _name(value: str) -> bytes:
    if not isinstance(value, str) or not value or "\0" in value:
        raise WireError("field names must be nonempty UTF-8 without NUL")
    try:
        return value.encode("utf-8")
    except UnicodeError as error:
        raise WireError("invalid UTF-8 field name") from error


def _encode(fields: Mapping[str, int | float | None], base_version: int, reset: bool) -> bytes:
    if type(base_version) is not int or not 0 <= base_version < 2**64:
        raise WireError("base version outside uint64")
    if len(fields) > MAX_FIELDS:
        raise WireError("too many fields")
    entries = []
    for name, value in fields.items():
        key = _name(name)
        if value is not None:
            if type(value) is int:
                if not -(2**31) <= value < 2**31:
                    raise WireError("value outside int32")
            elif type(value) is not float or not math.isfinite(value):
                raise WireError("expected int32 or finite double")
        entries.append((key, value))
    if sum(len(key) + 80 for key, _ in entries) + 80 > MAX_BYTES:
        raise WireError("fixture too large")
    data = bytearray(b"\0\0\0\0OODF")

    def align(width):
        data.extend(b"\0" * (-len(data) % width))

    def put(fmt, offset, *values):
        struct.pack_into("<" + fmt, data, offset, *values)

    def table(offsets, size):
        align(2)
        vt = len(data)
        data.extend(struct.pack("<" + "H" * (len(offsets) + 2), 4 + 2 * len(offsets), size, *offsets))
        align(8)
        result = len(data)
        data.extend(b"\0" * size)
        put("i", result, result - vt)
        return result

    def blob(raw, terminator=False):
        align(4)
        result = len(data)
        data.extend(struct.pack("<I", len(raw)))
        data.extend(raw)
        if terminator:
            data.append(0)
        return result

    # The union vector expands into two FlatBuffer fields (tags, tables), so
    # setFields is slot 4, not slot 3 of Delta.
    root = table([4, 8, 0, 0, 16 if entries else 0, 0, 0], 24)
    put("I", 0, root)
    put("B", root + 4, int(reset))
    put("Q", root + 8, base_version)
    if entries:
        align(4)
        vector = len(data)
        data.extend(struct.pack("<I", len(entries)))
        data.extend(b"\0" * (4 * len(entries)))
        put("I", root + 16, vector - root - 16)
        for index, (key, value) in enumerate(entries):
            record = table([8, 16, 20 if value is not None else 0, 0, 0, 24], 32)
            slot = vector + 4 + 4 * index
            put("I", slot, record - slot)
            put("Q", record + 8, 1)
            put("Q", record + 24, index)
            name = blob(key, True)
            put("I", record + 16, name - record - 16)
            if value is not None:
                encoded = blob(struct.pack("<Hi", 3, value) if type(value) is int else
                               struct.pack("<Hd", 9, value))
                put("I", record + 20, encoded - record - 20)
    return struct.pack("<4sIQ", b"OMPF", 1, 16) + bytes(data)


def empty_snapshot() -> bytes:
    """An empty authoritative reset: isDiff=true, baseVersion=0."""
    return _encode({}, 0, True)


def snapshot(fields: Mapping[str, int | float | None]) -> bytes:
    """A complete root-field snapshot (isDiff=true, baseVersion=0)."""
    return _encode(fields, 0, True)


def aggregate(fields: Mapping[str, int | float | None], base_version: int) -> bytes:
    """A root-field aggregate relative to a nonzero known base."""
    if type(base_version) is not int or not 0 < base_version < 2**64:
        raise WireError("aggregate requires nonzero uint64 base")
    return _encode(fields, base_version, True)


def update(fields: Mapping[str, int | float | None], base_version: int = 0) -> bytes:
    """An ordinary root-field update, in mapping iteration order."""
    return _encode(fields, base_version, False)


class _Reader:
    def __init__(self, data):
        self.data = data

    def span(self, offset, count):
        if offset < 0 or count < 0 or offset > len(self.data) - count:
            raise WireError("truncated or out-of-range FlatBuffer span")
        return self.data[offset:offset + count]

    def number(self, fmt, offset):
        return struct.unpack("<" + fmt, self.span(offset, struct.calcsize("<" + fmt)))[0]

    def indirect(self, slot):
        distance = self.number("I", slot)
        if distance < 4:
            raise WireError("invalid forward offset")
        target = slot + distance
        self.span(target, 4)
        return target

    def table(self, position, widths):
        if position % 4:
            raise WireError("unaligned table")
        distance = self.number("i", position)
        if not distance:
            raise WireError("zero vtable displacement")
        vt = position - distance  # Shared vtables may be after the table.
        size = self.number("H", vt)
        obj_size = self.number("H", vt + 2)
        if vt % 2 or size < 4 or size % 2 or obj_size < 4:
            raise WireError("invalid vtable shape")
        self.span(vt, size)
        self.span(position, obj_size)
        slots = []
        for i in range((size - 4) // 2):
            offset = self.number("H", vt + 4 + i * 2)
            if not offset:
                slots.append(None)
                continue
            if i >= len(widths) or offset < 4 or offset + widths[i] > obj_size:
                raise WireError("field outside supported table shape")
            slot = position + offset
            if slot % widths[i]:
                raise WireError("unaligned scalar field")
            slots.append(slot)
        return slots + [None] * (len(widths) - len(slots))

    def scalar(self, slots, index, fmt):
        return self.number(fmt, slots[index]) if slots[index] is not None else 0

    def vector(self, slot, width=1):
        if slot is None:
            return None
        position = self.indirect(slot)
        if position % 4:
            raise WireError("unaligned vector")
        count = self.number("I", position)
        self.span(position + 4, count * width)
        return position + 4, count


def decode_envelope(data: bytes) -> Delta:
    """Parse the supported subset, retaining field order and edit multiplicity."""
    if not isinstance(data, bytes) or not 24 <= len(data) <= MAX_BYTES:
        raise WireError("invalid envelope size or type")
    magic, count, offset = struct.unpack_from("<4sIQ", data)
    if magic != b"OMPF" or count != 1 or not 16 <= offset <= len(data) - 8:
        raise WireError("expected one bounded OMPF part")
    part = data[offset:]
    if part[4:8] != b"OODF":
        raise WireError("invalid OODF identifier")
    reader = _Reader(part)
    root = reader.indirect(0)
    slots = reader.table(root, [1, 8, 4, 4, 4, 4, 4])
    flag = reader.scalar(slots, 0, "B")
    if flag not in (0, 1):
        raise WireError("invalid diff flag")
    base = reader.scalar(slots, 1, "Q")
    if (slots[2] is None) != (slots[3] is None):
        raise WireError("command union vectors must be paired")
    for index, width in ((2, 1), (3, 4), (5, 4)):
        vector = reader.vector(slots[index], width)
        if vector is not None and vector[1]:
            raise WireError("commands and samples are outside the root-field oracle")
    if slots[6] is not None:
        raise WireError("sourceFormat is outside the root-field oracle")
    vector = reader.vector(slots[4], 4)
    if vector is not None and vector[1] > MAX_FIELDS:
        raise WireError("too many fields")
    fields = []
    if vector is not None:
        for index in range(vector[1]):
            record = reader.table(reader.indirect(vector[0] + 4 * index), [8, 4, 4, 8, 4, 8])
            if reader.scalar(record, 0, "Q") != 1:
                raise WireError("only root section fields are supported")
            if reader.scalar(record, 3, "Q") or reader.scalar(record, 4, "I"):
                raise WireError("external values are unsupported")
            key = reader.vector(record[1])
            if key is None or not key[1] or reader.number("B", key[0] + key[1]) != 0:
                raise WireError("missing or unterminated field name")
            try:
                name = reader.span(*key).decode("utf-8")
            except UnicodeError as error:
                raise WireError("invalid UTF-8 field name") from error
            _name(name)
            value = reader.vector(record[2])
            number = None
            if value is not None:
                if value[1] == 6 and reader.number("H", value[0]) == 3:
                    number = reader.number("i", value[0] + 2)
                elif value[1] == 10 and reader.number("H", value[0]) == 9:
                    number = reader.number("d", value[0] + 2)
                    if not math.isfinite(number):
                        raise WireError("nonfinite doubles are outside the root-field oracle")
                else:
                    raise WireError("expected scalar version-zero int32 or double serialization")
            fields.append(Field(name, number, reader.scalar(record, 5, "Q")))
    return Delta(bool(flag), base, tuple(fields))


def decode_update(data: bytes) -> dict[str, int | float | None]:
    delta = decode_envelope(data)
    if delta.is_diff:
        raise WireError("expected ordinary update")
    return {field.name: field.value for field in delta.fields}


class Model:
    """Independent commit oracle. Parse/sequence failures leave it unchanged.

    history retains (optional service sequence, Delta) per commit, including
    no-ops and equal-value writes; state alone cannot establish no retry/loss.
    """
    def __init__(self):
        self._state = {}
        self._history = []
        self._sequence = None

    @property
    def state(self):
        return dict(self._state)

    @property
    def history(self):
        return tuple(self._history)

    @property
    def commits(self):
        return len(self._history)

    @property
    def edits(self):
        return sum(len(delta.fields) for _, delta in self._history)

    def apply(self, data: bytes, sequence: int | None = None) -> dict[str, int | float]:
        delta = decode_envelope(data)
        if sequence is not None:
            if type(sequence) is not int or sequence < 0 or (self._sequence is not None and sequence <= self._sequence):
                raise WireError("service commit sequence must strictly increase")
        result = {} if delta.is_diff and delta.base_version == 0 else dict(self._state)
        for field in delta.fields:
            if field.value is None:
                result.pop(field.name, None)
            else:
                result[field.name] = field.value
        self._state = result
        self._history.append((sequence, delta))
        if sequence is not None:
            self._sequence = sequence
        return self.state
