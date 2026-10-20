# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent bounded structural OMPF profile, not a semantic graph validator.

Only struct and the published wire layout are used. All six command variants
are represented; supported values are bool, int32, finite double, scalar token,
specifier, variability, token/path vectors, path list operations, time samples and
field deletion, plus sourceFormat metadata on diffs. The value families Kit's
presence layer authors are supported too: string and string array, token array,
and the float, double array, matrix4d, vec2f, vec3d and vec4f families with
their scalar and array forms, held by exact IEEE-754 bits. Nested sample maps,
external values and other value families fail explicitly. Limits cover repeated
references as well as distinct storage.
"""
from dataclasses import dataclass
import math
import re
import struct

from scene_wire import _Reader, WireError

MAX_BYTES = 1024 * 1024
MAX_RECORDS = 4096
MAX_STRING_BYTES = 256 * 1024


class GraphWireError(ValueError):
    """Failure category: malformed input, unsupported profile, or quota limit."""
    def __init__(self, message, *, kind="malformed"):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class Command:
    tag: int
    section: int
    parent: int = 0
    name: str = ""
    spec_type: int = 0
    order: int = 0
    old_parent: int = 0
    child_list: int = 0
    children: tuple[int, ...] = ()


@dataclass(frozen=True)
class Token:
    value: str


@dataclass(frozen=True)
class Specifier:
    value: int


@dataclass(frozen=True)
class Variability:
    value: int


@dataclass(frozen=True)
class Paths:
    value: tuple[str, ...]


@dataclass(frozen=True)
class TimeSamples:
    value: tuple[tuple[float, object], ...] = ()


@dataclass(frozen=True)
class String:
    """A USD string (wire type 10). The profile holds UTF-8 text without NUL; other byte strings are valid wire
    values outside it."""
    value: str


@dataclass(frozen=True)
class Strings:
    """A USD string array (wire type 10 with the array flag)."""
    value: tuple[str, ...]


@dataclass(frozen=True)
class TokenArray:
    """A USD token array (wire type 11 with the array flag), a different value type from the token vector (type
    41) that child lists use."""
    value: tuple[str, ...]


ARRAY_FLAG = 0x80
# Wire type, component format and component count of each floating-point family. A scalar double stays a finite
# Python float, so only its array form is a Real.
REAL_FAMILIES = {'float': (8, 'f', 1), 'double': (9, 'd', 1), 'matrix4d': (15, 'd', 16),
                 'vec2f': (20, 'f', 2), 'vec3d': (23, 'd', 3), 'vec4f': (28, 'f', 4)}
_REAL_TAGS = {tag: family for family, (tag, _, _) in REAL_FAMILIES.items()}
_BITS = {'f': 'I', 'd': 'Q'}


@dataclass(frozen=True)
class Real:
    """A floating-point value held by the exact IEEE-754 bits of its components, row-major for a matrix and
    element after element for an array. Equality compares representation, as the wire does: -0.0 differs from
    0.0 and NaN payloads stay distinct."""
    family: str
    bits: tuple[int, ...]
    array: bool = False


def real(family, *components, array=False):
    """A Real from Python floats, each exactly representable in the family's component width."""
    if family not in REAL_FAMILIES or any(type(component) is not float for component in components):
        raise GraphWireError('real components must be floats of a known family')
    fmt = REAL_FAMILIES[family][1]
    try:
        raw = struct.pack('<' + fmt * len(components), *components)
    except (OverflowError, struct.error) as error:
        raise GraphWireError('real component outside its width') from error
    doubles = '<' + 'd' * len(components)
    if struct.pack(doubles, *struct.unpack('<' + fmt * len(components), raw)) != struct.pack(doubles, *components):
        raise GraphWireError('real component not exactly representable')
    return Real(family, struct.unpack('<' + _BITS[fmt] * len(components), raw), array)


def checked_real(value):
    """The value's wire tag, component format and width; the shape must be exact."""
    if type(value.family) is not str or value.family not in REAL_FAMILIES or type(value.array) is not bool:
        raise GraphWireError('invalid real family')
    tag, fmt, width = REAL_FAMILIES[value.family]
    if type(value.bits) is not tuple:
        raise GraphWireError('real requires an immutable tuple')
    if value.array:
        if len(value.bits) % width:
            raise GraphWireError('real array holds a partial element')
    elif value.family == 'double':
        raise GraphWireError('a scalar double is a Python float')
    elif len(value.bits) != width:
        raise GraphWireError('real scalar has the wrong component count')
    limit = 1 << (32 if fmt == 'f' else 64)
    if any(type(bits) is not int or not 0 <= bits < limit for bits in value.bits):
        raise GraphWireError('real component bits outside its width')
    return tag | (ARRAY_FLAG if value.array else 0), fmt, width


def checked_time(time):
    if type(time) is not float or math.isnan(time):
        raise GraphWireError('sample time must be a non-NaN double')
    return time


@dataclass(frozen=True)
class PathOp:
    explicit: bool = False
    explicit_items: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    prepended: tuple[str, ...] = ()
    appended: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    ordered: tuple[str, ...] = ()


PATH_OP_BUCKETS = ((2, 'explicit_items'), (4, 'added'), (32, 'prepended'),
                   (64, 'appended'), (8, 'deleted'), (16, 'ordered'))


def checked_path_op(value):
    if type(value.explicit) is not bool:
        raise GraphWireError('invalid explicit list mode')
    for bit, name in PATH_OP_BUCKETS:
        items = getattr(value, name)
        if type(items) is not tuple:
            raise GraphWireError('list bucket requires immutable tuple')
        if len(items) > MAX_RECORDS:
            raise GraphWireError('list bucket element bound', kind='limit')
        if items and ((bit == 2) != value.explicit):
            raise GraphWireError('list bucket incompatible with explicit mode')


_ABSOLUTE_PATH = re.compile(r'/(?:[A-Za-z_][A-Za-z_0-9]*)(?:/[A-Za-z_][A-Za-z_0-9]*)*(?:\.[A-Za-z_][A-Za-z_0-9]*(?::[A-Za-z_][A-Za-z_0-9]*)*)?\Z')


def checked_path(value):
    # Bounded independent grammar. Relative, nested-target, variant and Unicode
    # paths are valid USD families outside this profile, not malformed USD.
    if type(value) is not str:
        raise GraphWireError('path must be text')
    if not _ABSOLUTE_PATH.fullmatch(value):
        raise GraphWireError('path outside absolute ASCII graph profile', kind='unsupported')
    return value


def typed_value(value):
    """JSON projection preserves USD scalar types, including bool versus int."""
    if type(value) is TimeSamples:
        return dict(type='time_samples', value=[dict(time_bits=struct.unpack('<Q', struct.pack('<d', time))[0],
                    value=typed_value(sample)) for time, sample in value.value])
    if type(value) is PathOp:
        return dict(type='path_op', value=dict(explicit=value.explicit,
                    **{name: list(getattr(value, name)) for _, name in PATH_OP_BUCKETS}))
    if type(value) is Real:
        return dict(type=value.family + ('[]' if value.array else ''), value=list(value.bits))
    if type(value) in (String, Strings, TokenArray):
        kind = {String: 'string', Strings: 'string[]', TokenArray: 'token[]'}[type(value)]
        return dict(type=kind, value=value.value if type(value) is String else list(value.value))
    kind = {bool: 'bool', int: 'int32', float: 'double', tuple: 'tokens',
            Token: 'token', Specifier: 'specifier', Variability: 'variability', Paths: 'paths'}[type(value)]
    raw = list(value.value) if type(value) is Paths else value.value if type(value) in (Token, Specifier, Variability) else list(value) if type(value) is tuple else value
    return dict(type=kind, value=raw)


class _Erased:
    """A field value that erases as a present but empty value, the service's compacted-snapshot form."""
    def __repr__(self):
        return "ERASED"


ERASED = _Erased()


@dataclass(frozen=True)
class Field:
    section: int
    key: str
    value: (bool | int | float | Token | Specifier | Variability | Paths | PathOp | TimeSamples | tuple[str, ...] |
            String | Strings | TokenArray | Real | None)
    set_order: int = 0


@dataclass(frozen=True)
class Sample:
    section: int
    time: float
    value: object


@dataclass(frozen=True)
class Delta:
    is_diff: bool = False
    base_version: int = 0
    commands: tuple[Command, ...] = ()
    fields: tuple[Field, ...] = ()
    source_format: str = ""
    samples: tuple[Sample, ...] = ()


class _Budget:
    def __init__(self):
        self.records = MAX_RECORDS
        self.strings = MAX_STRING_BYTES

    def count(self, amount):
        if amount > self.records:
            raise GraphWireError("shared record/element limit exceeded", kind="limit")
        self.records -= amount

    def text(self, value):
        if type(value) is not str:
            raise GraphWireError("invalid UTF-8 string")
        if len(value) > self.strings:
            raise GraphWireError("shared string-byte limit exceeded", kind="limit")
        if "\0" in value:
            raise GraphWireError("invalid UTF-8 string")
        try:
            raw = value.encode("utf-8")
        except UnicodeError as error:
            raise GraphWireError("invalid UTF-8 string") from error
        self.raw_text(raw)
        return raw

    def string(self, raw):
        """A string value's bytes. The wire allows any bytes; text that is not UTF-8 or holds NUL is outside the
        profile, not malformed."""
        if len(raw) > self.strings:
            raise GraphWireError("shared string-byte limit exceeded", kind="limit")
        try:
            value = raw.decode("utf-8")
        except UnicodeError as error:
            raise GraphWireError("non-UTF-8 string outside graph profile", kind="unsupported") from error
        if "\0" in value:
            raise GraphWireError("string with NUL outside graph profile", kind="unsupported")
        self.strings -= len(raw)
        return value

    def string_text(self, value):
        if type(value) is not str:
            raise GraphWireError("string value must be text")
        try:
            raw = value.encode("utf-8")
        except UnicodeError as error:
            raise GraphWireError("string value is not encodable as UTF-8") from error
        self.string(raw)
        return raw

    def raw_text(self, raw):
        if len(raw) > self.strings:
            raise GraphWireError("shared string-byte limit exceeded", kind="limit")
        if b"\0" in raw:
            raise GraphWireError("invalid UTF-8 string")
        try:
            value = raw.decode("utf-8")
        except UnicodeError as error:
            raise GraphWireError("invalid UTF-8 string") from error
        self.strings -= len(raw)
        return value


def _uint(value, bits):
    if type(value) is not int or not 0 <= value < 1 << bits:
        raise GraphWireError(f"expected uint{bits}")
    return value


def _value(value, budget, *, in_samples=False):
    if value is None:
        return None
    if value is ERASED:
        return b""
    if type(value) is TimeSamples:
        if in_samples:
            raise GraphWireError('nested sample maps outside graph profile', kind='unsupported')
        if type(value.value) is not tuple:
            raise GraphWireError('sample map requires immutable tuple')
        budget.count(len(value.value))
        entries, times = [], set()
        for pair in value.value:
            if type(pair) is not tuple or len(pair) != 2:
                raise GraphWireError('invalid sample map entry')
            time, sample = pair
            checked_time(time)
            if time in times or sample is None:
                raise GraphWireError('duplicate sample time or empty sample value')
            times.add(time)
            entries.append((time, _value(sample, budget, in_samples=True)))
        return struct.pack('<HQ', 46, len(entries)) + b''.join(
            struct.pack('<d', time) + raw for time, raw in sorted(entries, key=lambda item: item[0]))
    if type(value) is bool:
        return struct.pack('<HB', 1, int(value))
    if type(value) is Token:
        raw = budget.text(value.value)
        return struct.pack('<HI', 11, len(raw)) + raw
    if type(value) in (Specifier, Variability):
        maximum, tag = (3, 42) if type(value) is Specifier else (2, 44)
        if type(value.value) is not int or not 0 <= value.value < maximum:
            raise GraphWireError('invalid USD enumeration')
        return struct.pack('<HI', tag, value.value)
    if type(value) is int:
        if not -(1 << 31) <= value < 1 << 31:
            raise GraphWireError("expected int32")
        return struct.pack("<Hi", 3, value)
    if type(value) is float:
        if not math.isfinite(value):
            raise GraphWireError("nonfinite double outside graph profile", kind="unsupported")
        return struct.pack("<Hd", 9, value)
    if type(value) in (tuple, Paths):
        items = value.value if type(value) is Paths else value
        if type(items) is not tuple:
            raise GraphWireError('vector requires an immutable tuple')
        budget.count(len(items))
        tokens = [budget.text(token) for token in items]
        if type(value) is Paths:
            for item in items:
                checked_path(item)
        return struct.pack("<HQ", 40 if type(value) is Paths else 41, len(tokens)) + b"".join(struct.pack("<I", len(t)) + t for t in tokens)
    if type(value) is PathOp:
        checked_path_op(value)
        flags, encoded = int(value.explicit), []
        for bit, name in PATH_OP_BUCKETS:
            items = getattr(value, name)
            if items:
                encoded.append(_value(Paths(items), budget)[2:])
                if len(set(items)) != len(items):
                    raise GraphWireError('duplicate path list item')
                flags |= bit
        return struct.pack('<HB', 34, flags) + b''.join(encoded)
    if type(value) is String:
        raw = budget.string_text(value.value)
        return struct.pack("<HI", 10, len(raw)) + raw
    if type(value) in (Strings, TokenArray):
        if type(value.value) is not tuple:
            raise GraphWireError('array requires an immutable tuple')
        budget.count(len(value.value))
        text = budget.string_text if type(value) is Strings else budget.text
        items = [text(item) for item in value.value]
        return struct.pack("<HQ", (10 if type(value) is Strings else 11) | ARRAY_FLAG, len(items)) + b"".join(
            struct.pack("<I", len(item)) + item for item in items)
    if type(value) is Real:
        tag, fmt, width = checked_real(value)
        bits = '<' + _BITS[fmt] * len(value.bits)
        if value.array:
            budget.count(len(value.bits) // width)
            return struct.pack("<HQ", tag, len(value.bits) // width) + struct.pack(bits, *value.bits)
        return struct.pack("<H", tag) + struct.pack(bits, *value.bits)
    raise GraphWireError("invalid field value type")


# Slot widths follow OmniObjectDelta.fbs; the union expands to paired vectors.
_COMMAND_WIDTHS = {1: (8, 8, 4, 1), 2: (8, 8), 3: (8, 8, 8, 4),
                   4: (8, 1, 4), 5: (8, 8, 4, 1, 4), 6: (8,)}


class _Writer:
    def __init__(self):
        self.data = bytearray(b"\0\0\0\0OODF")

    def append(self, raw):
        if len(raw) > MAX_BYTES - 16 - len(self.data):
            raise GraphWireError("encoded byte limit exceeded", kind="limit")
        self.data.extend(raw)

    def align(self, width, prefix=0):
        self.append(b"\0" * (-(len(self.data) + prefix) % width))

    def put(self, fmt, offset, *values):
        struct.pack_into("<" + fmt, self.data, offset, *values)

    def table(self, widths, present=None):
        offsets, size = [], 4
        present = set(range(len(widths))) if present is None else set(present)
        for index, width in enumerate(widths):
            size += -size % width
            offsets.append(size if index in present else 0)
            size += width
        self.align(2)
        vt = len(self.data)
        self.append(struct.pack("<" + "H" * (2 + len(offsets)), 4 + 2 * len(offsets), size, *offsets))
        self.align(8)
        table = len(self.data)
        self.append(b"\0" * size)
        self.put("i", table, table - vt)
        return table, [table + offset if offset else None for offset in offsets]

    def vector(self, count, width):
        self.align(max(4, width), 4)
        vector = len(self.data)
        self.append(struct.pack("<I", count))
        self.append(b"\0" * (count * width))
        return vector

    def blob(self, raw, terminated=False):
        self.align(4)
        vector = len(self.data)
        self.append(struct.pack("<I", len(raw)))
        self.append(raw)
        if terminated:
            self.append(b"\0")
        return vector

    def ref(self, slot, target):
        self.put("I", slot, target - slot)


def encode(delta: Delta) -> bytes:
    """Serialize this profile, retaining record order and multiplicity."""
    if not isinstance(delta, Delta) or type(delta.is_diff) is not bool:
        raise GraphWireError("expected Delta with boolean is_diff")
    _uint(delta.base_version, 64)
    if type(delta.commands) is not tuple or type(delta.fields) is not tuple or type(delta.samples) is not tuple:
        raise GraphWireError("records must be tuples")
    budget = _Budget()
    source_format = budget.text(delta.source_format)
    if source_format and not delta.is_diff:
        raise GraphWireError("sourceFormat on ordinary delta")
    budget.count(len(delta.commands) + len(delta.fields) + len(delta.samples))
    prepared_commands = []
    for command in delta.commands:
        if not isinstance(command, Command) or type(command.tag) is not int or command.tag not in _COMMAND_WIDTHS:
            raise GraphWireError("unsupported command tag")
        for value in (command.section, command.parent, command.old_parent):
            _uint(value, 64)
        _uint(command.spec_type, 8); _uint(command.order, 32); _uint(command.child_list, 8)
        if type(command.children) is not tuple:
            raise GraphWireError("children must be a tuple")
        budget.count(len(command.children))
        for child in command.children:
            _uint(child, 64)
        # Do not silently discard attributes absent from this union variant.
        used = {1: {"parent", "name", "spec_type"}, 2: {"parent"},
                3: {"parent", "old_parent", "name"}, 4: {"child_list", "children"},
                5: {"parent", "name", "spec_type", "order"}, 6: set()}[command.tag]
        defaults = {"parent": 0, "name": "", "spec_type": 0, "order": 0,
                    "old_parent": 0, "child_list": 0, "children": ()}
        if any(getattr(command, key) != default for key, default in defaults.items() if key not in used):
            raise GraphWireError("unused command attribute would be lost")
        prepared_commands.append((command, budget.text(command.name)))
    prepared_fields = []
    for field in delta.fields:
        if not isinstance(field, Field):
            raise GraphWireError("expected Field")
        _uint(field.section, 64); _uint(field.set_order, 64)
        prepared_fields.append((field, budget.text(field.key), _value(field.value, budget)))
    prepared_samples = []
    for sample in delta.samples:
        if not isinstance(sample, Sample):
            raise GraphWireError('expected Sample')
        _uint(sample.section, 64)
        if not sample.section:
            raise GraphWireError('invalid sample section')
        checked_time(sample.time)
        prepared_samples.append((sample, _value(sample.value, budget, in_samples=True)))
    writer = _Writer()
    present = ({0, 1} | ({2, 3} if delta.commands else set()) | ({4} if delta.fields else set()) |
               ({6} if source_format else set()) | ({5} if delta.samples else set()))
    root, slots = writer.table((1, 8, 4, 4, 4, 4, 4), present)
    writer.put("I", 0, root)
    writer.put("B", slots[0], delta.is_diff)
    writer.put("Q", slots[1], delta.base_version)
    if delta.commands:
        tags = writer.vector(len(delta.commands), 1)
        records = writer.vector(len(delta.commands), 4)
        writer.ref(slots[2], tags); writer.ref(slots[3], records)
        for index, (command, name) in enumerate(prepared_commands):
            writer.put("B", tags + 4 + index, command.tag)
            table, cs = writer.table(_COMMAND_WIDTHS[command.tag])
            writer.ref(records + 4 + 4 * index, table)
            if command.tag in (1, 5):
                writer.put("Q", cs[0], command.parent); writer.put("Q", cs[1], command.section)
                writer.ref(cs[2], writer.blob(name, True)); writer.put("B", cs[3], command.spec_type)
                if command.tag == 5:
                    writer.put("I", cs[4], command.order)
            elif command.tag == 2:
                writer.put("Q", cs[0], command.parent); writer.put("Q", cs[1], command.section)
            elif command.tag == 3:
                writer.put("Q", cs[0], command.old_parent); writer.put("Q", cs[1], command.parent)
                writer.put("Q", cs[2], command.section); writer.ref(cs[3], writer.blob(name, True))
            else:
                writer.put("Q", cs[0], command.section)
                if command.tag == 4:
                    writer.put("B", cs[1], command.child_list)
                    children = writer.vector(len(command.children), 8)
                    writer.ref(cs[2], children)
                    for i, child in enumerate(command.children):
                        writer.put("Q", children + 4 + 8 * i, child)
    if delta.fields:
        records = writer.vector(len(delta.fields), 4)
        writer.ref(slots[4], records)
        for index, (field, key, value) in enumerate(prepared_fields):
            table, fs = writer.table((8, 4, 4, 8, 4, 8), {0, 1, 5} | ({2} if value is not None else set()))
            writer.ref(records + 4 + 4 * index, table)
            writer.put("Q", fs[0], field.section); writer.put("Q", fs[5], field.set_order)
            writer.ref(fs[1], writer.blob(key, True))
            if value is not None:
                writer.ref(fs[2], writer.blob(value))
    if delta.samples:
        records = writer.vector(len(delta.samples), 4)
        writer.ref(slots[5], records)
        for index, (sample, value) in enumerate(prepared_samples):
            table, ss = writer.table((8, 8, 4, 8, 4), {0, 1} | ({2} if value is not None else set()))
            writer.ref(records + 4 + 4 * index, table)
            writer.put('Q', ss[0], sample.section); writer.put('d', ss[1], sample.time)
            if value is not None:
                writer.ref(ss[2], writer.blob(value))
    if source_format:
        writer.ref(slots[6], writer.blob(source_format, True))
    return struct.pack("<4sIQ", b"OMPF", 1, 16) + bytes(writer.data)


def _table(reader, position, widths):
    # _Reader permits trailing absent schema slots; this closed profile also
    # bounds traversal work when thousands of records alias one large vtable.
    distance = reader.number("i", position)
    size = reader.number("H", position - distance)
    if size > 4 + 2 * len(widths):
        raise GraphWireError("table exceeds supported schema shape")
    return reader.table(position, widths)


def _text(reader, slot, budget):
    vector = reader.vector(slot)
    if vector is None:
        return ""
    if reader.number("B", vector[0] + vector[1]) != 0:
        raise GraphWireError("unterminated string")
    return budget.raw_text(reader.span(*vector))


def _decode_value(raw, budget):
    if not raw:
        return None  # An explicit empty top-level value is an erasure.
    value, position = _read_value(raw, 0, budget)
    if position != len(raw):
        raise GraphWireError('trailing value bytes')
    return value


def _read_value(raw, position, budget, *, in_samples=False):
    def scalar(fmt):
        nonlocal position
        width = struct.calcsize('<' + fmt)
        if position > len(raw) - width:
            raise GraphWireError('truncated value')
        value = struct.unpack_from('<' + fmt, raw, position)[0]
        position += width
        return value

    tag = scalar('H')
    if tag == 1:
        value = scalar('B')
        if value > 1:
            raise GraphWireError('malformed boolean')
        return bool(value), position
    if tag in (42, 44):
        maximum, value_type = (3, Specifier) if tag == 42 else (2, Variability)
        value = scalar('I')
        if value >= maximum:
            raise GraphWireError('invalid USD enumeration')
        return value_type(value), position
    if tag == 3:
        return scalar('i'), position
    if tag == 9:
        value = scalar('d')
        if not math.isfinite(value):
            raise GraphWireError('nonfinite double outside graph profile', kind='unsupported')
        return value, position
    if tag == 11:
        length = scalar('I')
        if length > len(raw) - position:
            raise GraphWireError('truncated scalar token')
        return Token(budget.raw_text(raw[position:position + length])), position + length
    if tag == 34:
        flags = scalar('B')
        if flags & 128 or flags & 2 and not flags & 1 or flags & 1 and flags & 124:
            raise GraphWireError('invalid path list flags')
        buckets = {}
        for bit, name in PATH_OP_BUCKETS:
            if flags & bit:
                values, position = _read_strings(raw, position, budget)
                values = tuple(checked_path(value) for value in values)
                if len(set(values)) != len(values):
                    raise GraphWireError('duplicate path list item')
                buckets[name] = values
        return PathOp(bool(flags & 1), **buckets), position
    if tag in (40, 41):
        values, position = _read_strings(raw, position, budget)
        return (Paths(tuple(checked_path(value) for value in values)) if tag == 40 else values), position
    if tag == 10:
        length = scalar('I')
        if length > len(raw) - position:
            raise GraphWireError('truncated string')
        return String(budget.string(raw[position:position + length])), position + length
    if tag in (10 | ARRAY_FLAG, 11 | ARRAY_FLAG):
        strings = tag == 10 | ARRAY_FLAG
        values, position = _read_strings(raw, position, budget, budget.string if strings else budget.raw_text)
        return (Strings if strings else TokenArray)(values), position
    family = _REAL_TAGS.get(tag & ~ARRAY_FLAG)
    if family is not None and (tag & ARRAY_FLAG or family != 'double'):
        _, fmt, width = REAL_FAMILIES[family]
        size, count = width * struct.calcsize(fmt), 1
        if tag & ARRAY_FLAG:
            count = scalar('Q')
            if count > (len(raw) - position) // size:
                raise GraphWireError('real array count exceeds content')
            budget.count(count)
        if count * size > len(raw) - position:
            raise GraphWireError('truncated real value')
        bits = struct.unpack_from('<' + _BITS[fmt] * (count * width), raw, position)
        return Real(family, bits, bool(tag & ARRAY_FLAG)), position + count * size
    if tag == 46:
        if in_samples:
            raise GraphWireError('nested sample maps outside graph profile', kind='unsupported')
        count = scalar('Q')
        budget.count(count)
        entries, times = [], set()
        for _ in range(count):
            time = checked_time(scalar('d'))
            if time in times:
                raise GraphWireError('duplicate sample time')
            times.add(time)
            value, position = _read_value(raw, position, budget, in_samples=True)
            entries.append((time, value))
        return TimeSamples(tuple(sorted(entries, key=lambda item: item[0]))), position
    # Excluded families are not evidence of production wire invalidity.
    raise GraphWireError('value type/version outside graph profile', kind='unsupported')


def _read_strings(raw, position, budget, text=None):
    text = text or budget.raw_text
    if position > len(raw) - 8:
        raise GraphWireError('truncated vector count')
    count = struct.unpack_from("<Q", raw, position)[0]
    # Each item needs at least its 4-byte length, so a larger count is malformed before it is charged.
    if count > (len(raw) - position - 8) // 4:
        raise GraphWireError('vector count exceeds content')
    budget.count(count)
    position, values = position + 8, []
    for _ in range(count):
        if position > len(raw) - 4:
            raise GraphWireError("truncated token length")
        length = struct.unpack_from("<I", raw, position)[0]
        position += 4
        if length > len(raw) - position:
            raise GraphWireError("truncated token")
        values.append(text(raw[position:position + length]))
        position += length
    return tuple(values), position


def _decode(data):
    if type(data) is not bytes:
        raise GraphWireError("invalid envelope type")
    if len(data) > MAX_BYTES:
        raise GraphWireError("encoded byte limit exceeded", kind="limit")
    if len(data) < 24:
        raise GraphWireError("truncated envelope")
    magic, count, offset = struct.unpack_from("<4sIQ", data)
    if magic != b"OMPF" or count == 0 or count > (len(data) - 8) // 8:
        raise GraphWireError("invalid OMPF framing")
    if count > MAX_RECORDS:
        raise GraphWireError("part-count limit exceeded", kind="limit")
    offsets = struct.unpack_from("<" + "Q" * count, data, 8)
    previous = 8 + 8 * count - 1
    for position in offsets:
        if not previous < position < len(data):
            raise GraphWireError("invalid OMPF part offset")
        previous = position
    if count != 1:
        raise GraphWireError("multiple parts outside graph profile", kind="unsupported")
    if offset > len(data) - 8:
        raise GraphWireError("truncated OODF part")
    part = data[offset:]
    if part[4:8] != b"OODF":
        raise GraphWireError("invalid OODF identifier")
    reader, budget = _Reader(part), _Budget()
    slots = _table(reader, reader.indirect(0), (1, 8, 4, 4, 4, 4, 4))
    flag = reader.scalar(slots, 0, "B")
    if flag not in (0, 1):
        raise GraphWireError("invalid diff flag")
    if (slots[2] is None) != (slots[3] is None):
        raise GraphWireError("unpaired command union vectors")
    tags, records = reader.vector(slots[2]), reader.vector(slots[3], 4)
    if tags is not None and tags[1] != records[1]:
        raise GraphWireError("mismatched command union counts")
    sample_records = reader.vector(slots[5], 4)
    source_format = _text(reader, slots[6], budget)
    if source_format and not flag:
        raise GraphWireError("sourceFormat on ordinary delta")
    field_records = reader.vector(slots[4], 4)
    budget.count((tags[1] if tags else 0) + (field_records[1] if field_records else 0) +
                 (sample_records[1] if sample_records else 0))
    commands, fields = [], []
    for index in range(tags[1] if tags else 0):
        tag = reader.number("B", tags[0] + index)
        if tag not in _COMMAND_WIDTHS:
            raise GraphWireError("unsupported command tag")
        cs = _table(reader, reader.indirect(records[0] + 4 * index), _COMMAND_WIDTHS[tag])
        scalar = lambda i, fmt="Q": reader.scalar(cs, i, fmt)
        if tag in (1, 5):
            command = Command(tag, scalar(1), parent=scalar(0), name=_text(reader, cs[2], budget),
                              spec_type=scalar(3, "B"), order=scalar(4, "I") if tag == 5 else 0)
        elif tag == 2:
            command = Command(tag, scalar(1), parent=scalar(0))
        elif tag == 3:
            command = Command(tag, scalar(2), old_parent=scalar(0), parent=scalar(1), name=_text(reader, cs[3], budget))
        elif tag == 4:
            vector = reader.vector(cs[2], 8)
            size = vector[1] if vector else 0
            budget.count(size)
            if size and vector[0] % 8:
                raise GraphWireError("unaligned section-ID vector")
            children = tuple(reader.number("Q", vector[0] + 8 * i) for i in range(size))
            command = Command(tag, scalar(0), child_list=scalar(1, "B"), children=children)
            if not command.section or command.child_list >= 3 or vector is None:
                raise GraphWireError("invalid child reorder")
            # Any ID and any repeat decode: the list is an order, applied by ignoring IDs that are not children
            # and repeats after an ID's first occurrence, or rejected by a service that stores it verbatim.
        else:
            command = Command(tag, scalar(0))
        commands.append(command)
    for index in range(field_records[1] if field_records else 0):
        fs = _table(reader, reader.indirect(field_records[0] + 4 * index), (8, 4, 4, 8, 4, 8))
        if reader.scalar(fs, 3, "Q") or reader.scalar(fs, 4, "I"):
            raise GraphWireError("external values outside graph profile", kind="unsupported")
        key = _text(reader, fs[1], budget)
        value = reader.vector(fs[2])
        fields.append(Field(reader.scalar(fs, 0, "Q"), key,
                            _decode_value(reader.span(*value), budget) if value is not None else None,
                            reader.scalar(fs, 5, "Q")))
    samples = []
    for index in range(sample_records[1] if sample_records else 0):
        ss = _table(reader, reader.indirect(sample_records[0] + 4 * index), (8, 8, 4, 8, 4))
        if reader.scalar(ss, 3, 'Q') or reader.scalar(ss, 4, 'I'):
            raise GraphWireError('external values outside graph profile', kind='unsupported')
        section = reader.scalar(ss, 0, 'Q')
        time = checked_time(float(reader.scalar(ss, 1, 'd')))
        if not section:
            raise GraphWireError('invalid sample section')
        value = reader.vector(ss[2])
        value = _decode_value(reader.span(*value), budget) if value is not None else None
        if type(value) is TimeSamples:
            raise GraphWireError('nested sample maps outside graph profile', kind='unsupported')
        samples.append(Sample(section, time, value))
    return Delta(bool(flag), reader.scalar(slots, 1, "Q"), tuple(commands), tuple(fields), source_format, tuple(samples))


def entry_counts(data: bytes):
    """Count a delta's entries as the live edit service does, before decoding it.

    Returns its struct commands, set fields and time samples, the FlatBuffers
    tables the service's verifier counts besides the root, and the length of
    every child reorder's children list, a scalar vector outside that count.
    Framing this profile cannot read returns None, and decode() rejects it.
    """
    try:
        if type(data) is not bytes or len(data) < 24:
            return None
        magic, count, offset = struct.unpack_from("<4sIQ", data)
        if magic != b"OMPF" or count != 1 or offset > len(data) - 8:
            return None
        part = data[offset:]
        if part[4:8] != b"OODF":
            return None
        reader = _Reader(part)
        slots = _table(reader, reader.indirect(0), (1, 8, 4, 4, 4, 4, 4))
        tags, records = reader.vector(slots[2]), reader.vector(slots[3], 4)
        fields, samples = reader.vector(slots[4], 4), reader.vector(slots[5], 4)
        commands = records[1] if records else 0
        if (tags[1] if tags else 0) != commands:
            return None
        children = []
        for index in range(commands):
            if reader.number("B", tags[0] + index) == 4:
                table = _table(reader, reader.indirect(records[0] + 4 * index), _COMMAND_WIDTHS[4])
                vector = reader.vector(table[2], 8)
                children.append(vector[1] if vector else 0)
        return commands, fields[1] if fields else 0, samples[1] if samples else 0, tuple(children)
    except (WireError, GraphWireError, struct.error):
        return None


def decode(data: bytes) -> Delta:
    """Decode structural records without guessing their graph semantics."""
    try:
        return _decode(data)
    except WireError as error:
        raise GraphWireError(str(error)) from error
