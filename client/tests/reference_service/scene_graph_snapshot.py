# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent bounded reset/diff projection for the reference graph profile.

Inputs are complete model states, not edit logs. A successful result reproduces
that exact state; it neither infers operation identities nor authorizes local
structural admission. No USD, generated FlatBuffers or production code is used.
"""
import math
import struct

import scene_graph_model as graph
import scene_graph_wire as wire
from scene_graph_wire import Command, Field, Delta

_KEYS = {"id", "parent", "type", "name", "order", "path", "fields"}
_METADATA = ("parent", "type", "name", "order")


def _fail(message, kind="malformed"):
    raise graph.GraphError(message, kind)


def _uint(value, bits):
    if type(value) is not int or not 0 <= value < 1 << bits:
        _fail(f"state scalar outside uint{bits}")


def _value_key(value):
    if type(value) is wire.TimeSamples:
        return 'time_samples', tuple((struct.pack('<d', time), _value_key(sample)) for time, sample in value.value)
    # A Real compares its component bits, the representation the wire carries.
    if type(value) in (bool, wire.Token, wire.Specifier, wire.Variability, wire.Paths, wire.PathOp,
                       wire.String, wire.Strings, wire.TokenArray, wire.Real):
        return type(value).__name__, value
    if type(value) is int:
        return "int32", value
    if type(value) is float:
        return "double", struct.pack("<d", value)
    return "tokens", value


def _signature(state):
    return tuple((node["id"], *(node[key] for key in _METADATA), node["path"],
                  tuple((key, _value_key(value)) for key, value in sorted(node["fields"].items())))
                 for node in state)


def _checked_state(state):
    if type(state) is not list:
        _fail("complete graph state must be a list")
    if len(state) > min(graph.MAX_SECTIONS, wire.MAX_RECORDS):
        _fail("state section/record limit exceeded", "limit")
    records, fields, strings, paths = len(state), 0, 0, 0
    sample_budget = wire._Budget()
    nodes = {}

    def text(value, path=False):
        nonlocal strings, paths
        if type(value) is not str:
            _fail("state metadata must be strings")
        remaining = graph.MAX_PATH_BYTES - paths if path else wire.MAX_STRING_BYTES - strings
        if len(value) > remaining:
            _fail("state string/path limit exceeded", "limit")
        try:
            encoded = value.encode("utf-8")
        except UnicodeError as error:
            raise graph.GraphError("invalid state UTF-8", "malformed") from error
        if len(encoded) > remaining:
            _fail("state string/path limit exceeded", "limit")
        if "\0" in value:
            _fail("state string contains NUL")
        if path:
            paths += len(encoded)
        else:
            strings += len(encoded)

    for node in state:
        if type(node) is not dict or node.keys() != _KEYS:
            _fail("incomplete or unknown state node fields")
        for key, bits in (("id", 64), ("parent", 64), ("type", 8), ("order", 32)):
            _uint(node[key], bits)
        if node["id"] in nodes:
            _fail("duplicate state identity")
        text(node["name"]); text(node["path"], path=True)
        if type(node["fields"]) is not dict:
            _fail("state fields must be a dictionary")
        fields += len(node["fields"])
        records += len(node["fields"])
        if fields > graph.MAX_FIELDS or records > wire.MAX_RECORDS:
            _fail("state field/record limit exceeded", "limit")
        for key, value in node["fields"].items():
            text(key)
            if type(value) in (wire.TimeSamples, wire.String, wire.Strings, wire.TokenArray, wire.Real):
                # Reuse the independent value traversal with the remaining
                # aggregate budgets, including every sample, array and string.
                sample_budget.records = wire.MAX_RECORDS - records
                sample_budget.strings = wire.MAX_STRING_BYTES - strings
                try:
                    wire._value(value, sample_budget)
                except wire.GraphWireError as error:
                    raise graph.GraphError(str(error), error.kind) from error
                records = wire.MAX_RECORDS - sample_budget.records
                strings = wire.MAX_STRING_BYTES - sample_budget.strings
            elif type(value) is bool:
                pass
            elif type(value) is wire.Token:
                text(value.value)
            elif type(value) in (wire.Specifier, wire.Variability):
                maximum = 3 if type(value) is wire.Specifier else 2
                if type(value.value) is not int or not 0 <= value.value < maximum:
                    _fail('invalid state USD enumeration')
            elif type(value) is int:
                if not -(1 << 31) <= value < 1 << 31:
                    _fail("state value outside int32")
            elif type(value) is float:
                if not math.isfinite(value):
                    _fail("nonfinite state value outside profile", "unsupported")
            elif type(value) in (tuple, wire.Paths, wire.PathOp):
                if type(value) is wire.PathOp:
                    try:
                        wire.checked_path_op(value)
                    except wire.GraphWireError as error:
                        raise graph.GraphError(str(error), error.kind) from error
                    items = tuple(item for _, name in wire.PATH_OP_BUCKETS for item in getattr(value, name))
                else:
                    items = value.value if type(value) is wire.Paths else value
                if type(items) is not tuple:
                    _fail('state vector requires immutable tuple')
                records += len(items)
                if records > wire.MAX_RECORDS:
                    _fail("state element limit exceeded", "limit")
                for token in items:
                    text(token)
                    if type(value) in (wire.Paths, wire.PathOp):
                        try:
                            wire.checked_path(token)
                        except wire.GraphWireError as error:
                            raise graph.GraphError(str(error), error.kind) from error
            elif value is None:
                _fail("stored state cannot contain deletion markers")
            else:
                _fail("state value family outside profile", "unsupported")
        nodes[node["id"]] = dict(node, fields=dict(node["fields"]))
    return [nodes[identity] for identity in sorted(nodes)]


def _encode(delta):
    try:
        return wire.encode(delta)
    except wire.GraphWireError as error:
        raise graph.GraphError(str(error), error.kind) from error


def _change(node):
    return Command(5, node["id"], parent=node["parent"], name=node["name"],
                   spec_type=node["type"], order=node["order"])


def _fields(node, erased):
    """A node's fields in key order, with each erased key it lacks as an ERASED field."""
    keys = set(node["fields"]) | {key for section, key in erased if section == node["id"]}
    return tuple(Field(node["id"], key, node["fields"].get(key, wire.ERASED)) for key in sorted(keys))


def _validated(state, source_format, erased=frozenset()):
    checked = _checked_state(state)
    orders = [(node["parent"], node["order"]) for node in checked if node["id"] != 1]
    if len(set(orders)) != len(orders):
        _fail("ambiguous sibling order", "unsupported")
    # The service emits each section's index among its parent's children as its order.
    expected = graph.ranked(checked)
    data = _encode(Delta(True, 0, tuple(_change(node) for node in expected),
                         tuple(field for node in expected for field in _fields(node, erased)), source_format))
    model = graph.Model()
    actual = model.apply(data)
    if _signature(actual) != _signature(expected) or model.source_format != source_format:
        # Reconstructed paths, child membership/order and field presence must
        # agree exactly; never silently normalize a caller's invalid state.
        _fail("state differs from its complete graph reconstruction")
    return expected, model, data


def snapshot(state: list[dict], *, source_format: str = "", erased: frozenset = frozenset()) -> bytes:
    """Encode a complete reset, including root, every field and source format.

    `erased` holds (section, key) pairs the snapshot keeps as ERASED fields, as the service's compacted snapshot
    keeps a field an update erased.
    """
    return _validated(state, source_format, erased)[2]


def difference(before: list[dict], after: list[dict], base_version: int, *,
               before_source_format: str = "", after_source_format: str = "") -> bytes:
    """Encode a final-state diff with exact typed field/presence changes.

    Deletes are nonrecursive DiffDelete records. Endpoints must be individually
    valid and the model must accept their transformation; otherwise no bytes
    are returned and neither input is changed. An empty diff sourceFormat means
    preserve, so clearing a nonempty source format requires a full reset.
    """
    _uint(base_version, 64)
    if not base_version:
        _fail("difference requires a nonzero base version")
    old, model, _ = _validated(before, before_source_format)
    new, _, _ = _validated(after, after_source_format)
    if before_source_format and not after_source_format:
        _fail("clearing sourceFormat requires a full reset", "unsupported")
    old_nodes, new_nodes = ({node["id"]: node for node in state} for state in (old, new))
    commands = [Command(6, identity) for identity in sorted(old_nodes.keys() - new_nodes.keys())]
    commands += [_change(node) for node in new
                 if node["id"] not in old_nodes or
                 any(old_nodes[node["id"]][key] != node[key] for key in _METADATA)]
    fields = []
    for node in new:
        old_fields = old_nodes.get(node["id"], {}).get("fields", {})
        for key in sorted(old_fields.keys() | node["fields"].keys()):
            if key not in node["fields"]:
                fields.append(Field(node["id"], key, None))
            elif key not in old_fields or _value_key(old_fields[key]) != _value_key(node["fields"][key]):
                fields.append(Field(node["id"], key, node["fields"][key]))
    source_format = after_source_format if after_source_format != before_source_format else ""
    data = _encode(Delta(True, base_version, tuple(commands), tuple(fields), source_format))
    if _signature(model.apply(data)) != _signature(new) or model.source_format != after_source_format:
        _fail("diff does not reproduce exact final graph", "unsupported")
    return data


class Versions:
    """Per-row change sequences, like the Live Edit service's materialized layer state.

    `sections` and `fields` hold the sequence that last created or changed each
    live section's metadata and each live field. `created` holds the sequence
    that created each live identity; `retired` holds the (created, deleted)
    sequence pairs of every retired incarnation, the service's tombstones.
    `cleared` holds the sequence that removed each field, including every field
    of a retired section, so a re-created identity can retract its earlier
    incarnation's fields.
    """
    __slots__ = ("sections", "fields", "created", "retired", "cleared", "source_format")

    def __init__(self, sections=None, fields=None, created=None, retired=None, cleared=None, source_format=0):
        self.sections, self.fields = dict(sections or {}), dict(fields or {})
        self.created, self.retired = dict(created or {}), dict(retired or {})
        self.cleared = dict(cleared or {})
        self.source_format = source_format

    def advanced(self, before, after, sequence, *, before_source_format="", after_source_format=""):
        """Return the record after one commit took `before` to `after` at `sequence`."""
        _uint(sequence, 64)
        if not sequence:
            _fail("versions require a nonzero sequence")
        result = Versions(self.sections, self.fields, self.created, self.retired, self.cleared, self.source_format)
        old = {node["id"]: node for node in before}
        new = {node["id"]: node for node in after}
        for identity in old.keys() - new.keys():
            result.sections.pop(identity, None)
            first = result.created.pop(identity, sequence)
            result.retired[identity] = result.retired.get(identity, ()) + ((first, sequence),)
            for key in old[identity]["fields"]:
                result.fields.pop((identity, key), None)
                result.cleared[identity, key] = sequence
        for identity, node in new.items():
            previous = old.get(identity)
            if previous is None:
                result.created[identity] = sequence
            if previous is None or any(previous[key] != node[key] for key in _METADATA):
                result.sections[identity] = sequence
            old_fields = previous["fields"] if previous else {}
            for key in old_fields.keys() | node["fields"].keys():
                if key not in node["fields"]:
                    result.fields.pop((identity, key), None)
                    result.cleared[identity, key] = sequence
                elif key not in old_fields or _value_key(old_fields[key]) != _value_key(node["fields"][key]):
                    result.fields[identity, key] = sequence
                    result.cleared.pop((identity, key), None)
        if after_source_format != before_source_format:
            result.source_format = sequence
        return result


def coalesced(after: list[dict], versions: Versions, base_version: int, *, source_format: str = "") -> bytes:
    """Encode the service's catch-up diff from `base_version` to `after`.

    Like the service's coalesced builder it needs no edit log. It retracts each
    retired identity a subscriber at the base held, that is one with an
    incarnation created at or before the base and deleted after it; identities
    created and retired after the base are dropped, never retracted. It
    restates every section and field changed after the base. Applied to any
    complete state at `base_version` it reproduces `after`.
    """
    _uint(base_version, 64)
    if not base_version:
        _fail("coalesced diff requires a nonzero base version")
    new, _, _ = _validated(after, source_format)
    live = {node["id"] for node in new}
    commands = [Command(6, identity) for identity, lives in sorted(versions.retired.items())
                if identity not in live and any(first <= base_version < last for first, last in lives)]
    commands += [_change(node) for node in new if versions.sections.get(node["id"], 0) > base_version]
    changed = {}
    for (identity, key), sequence in versions.fields.items():
        if sequence > base_version and identity in live:
            changed.setdefault(identity, set()).add(key)
    for (identity, key), sequence in versions.cleared.items():
        if sequence > base_version and identity in live:
            changed.setdefault(identity, set()).add(key)
    fields = [Field(node["id"], key, node["fields"].get(key))
              for node in new for key in sorted(changed.get(node["id"], ()))]
    if versions.source_format > base_version and not source_format:
        _fail("clearing sourceFormat requires a full reset", "unsupported")
    delta_format = source_format if versions.source_format > base_version else ""
    return _encode(Delta(True, base_version, tuple(commands), tuple(fields), delta_format))
