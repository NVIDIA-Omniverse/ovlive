#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse the native adversarial fixture with an independent standard JSON reader."""
import json
import subprocess
import sys


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f"duplicate JSON key: {key!r}"
        result[key] = value
    return result


output = subprocess.check_output([sys.argv[1], "--emit"], timeout=20)
lines = output.split(b"\n")
assert len(lines) == 3 and lines[-1] == b"", "expected exactly two terminated records"
assert all(32 <= byte < 127 for line in lines[:-1] for byte in line), "unescaped output byte"
record = json.loads(lines[0], object_pairs_hook=unique_object)
assert record["text_encoding"] == "escaped_bytes"
assert record["event_name"] == 'event"\\\n\x80'
assert record["timestamp_ms"] == 2**64 - 1
assert record["field_count"] == 4
assert record["fields"] == {
    "unsigned": {"kind": 1, "value": 2**64 - 1},
    "signed": {"kind": 2, "value": -(2**63)},
    'string"\n': {"kind": 3, "value": 'quote"slash\\\n\r\t\x01\x7f\xff'},
    "boolean": {"kind": 4, "value": True},
}
assert type(record["fields"]["unsigned"]["value"]) is int
assert type(record["fields"]["signed"]["value"]) is int
assert type(record["fields"]["boolean"]["value"]) is bool
assert lines[1].startswith(b'WARN "event')
print("independent JSON parser preserved integer extrema, field kinds and escaped bytes")
