#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline, bounded CopyDiagnostics validation. Never fetch schemas or print input data."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / 'tests/contracts/diagnostic-snapshot.schema.json'
CATALOG = ROOT / 'tests/contracts/diagnostic-catalog.json'
PINS = {'jsonschema': '4.26.0', 'referencing': '0.37.0', 'attrs': '26.1.0',
        'jsonschema-specifications': '2025.9.1', 'rpds-py': '0.30.0', 'typing-extensions': '4.15.0'}
MAX_BYTES = 16 * 1024 * 1024
MAX_DEPTH = 64
MAX_NODES = 1_000_000
MODULES = {'jsonschema': 'jsonschema', 'referencing': 'referencing', 'attrs': 'attrs',
           'attr': 'attrs', 'jsonschema_specifications': 'jsonschema-specifications',
           'rpds': 'rpds-py', 'typing_extensions': 'typing-extensions'}
_import_cache = None


def prepare_imports():
    global _import_cache
    if _import_cache is None:
        if any(name.split('.')[0] in MODULES for name in sys.modules):
            raise RuntimeError('validator dependencies were imported before cache isolation')
        # -B alone still READS existing bytecode. A fresh private prefix plus
        # disabled writes ensures dependencies execute inventoried source/DSOs.
        _import_cache = tempfile.TemporaryDirectory(prefix='live-schema-imports-')
        sys.dont_write_bytecode = True
        sys.pycache_prefix = _import_cache.name


class InvalidSnapshot(ValueError):
    """Messages are static classifications, never values from the input."""


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidSnapshot('duplicate object key')
        result[key] = value
    return result


def reject_constant(_):
    raise InvalidSnapshot('nonfinite number')


def parse_number(token):
    if len(token) > 128:
        raise InvalidSnapshot('numeric token budget exceeded')
    return Decimal(token)


def parse_integer(token):
    if len(token) > 128:
        raise InvalidSnapshot('numeric token budget exceeded')
    return int(token)


def load_document(content: bytes, *, max_bytes=MAX_BYTES):
    if len(content) > max_bytes:
        raise InvalidSnapshot('document byte budget exceeded')
    try:
        text = content.decode('utf-8', errors='strict')
        # Bound container depth and a conservative node-allocation estimate BEFORE
        # json.loads allocates containers. String contents do not count as syntax.
        depth = 0
        allocations = 1
        quoted = escaped = False
        for c in text:
            if quoted:
                if escaped:
                    escaped = False
                elif c == '\\':
                    escaped = True
                elif c == '"':
                    quoted = False
            elif c == '"':
                quoted = True
            elif c in '[{':
                depth += 1
                allocations += 1
                if depth > MAX_DEPTH:
                    raise InvalidSnapshot('document depth budget exceeded')
            elif c in ']}':
                depth -= 1
            elif c in ',:':
                allocations += 1
            if allocations > MAX_NODES:
                raise InvalidSnapshot('document node budget exceeded')
        return json.loads(text, object_pairs_hook=unique_object, parse_float=parse_number, parse_int=parse_integer,
                          parse_constant=reject_constant)
    except InvalidSnapshot:
        raise
    except (UnicodeError, ValueError, RecursionError, ArithmeticError):
        raise InvalidSnapshot('invalid JSON encoding or syntax') from None


def identity():
    prepare_imports()
    packages = {}
    origins = {}
    for name, version in PINS.items():
        distribution = importlib.metadata.distribution(name)
        if distribution.version != version:
            raise RuntimeError('diagnostic validator dependency version mismatch: ' + name)
        digest = hashlib.sha256()
        files = [f for f in distribution.files or [] if not str(f).endswith(('.pyc', '.pyo'))]
        if not files:
            raise RuntimeError('diagnostic validator dependency has no file inventory: ' + name)
        for entry in sorted(files, key=str):
            path = Path(distribution.locate_file(entry)).resolve()
            digest.update(str(entry).encode() + b'\0' + path.read_bytes())
            origins[path] = name
        packages[name] = {'version': version, 'files_sha256': digest.hexdigest()}
    for module in MODULES:
        importlib.import_module(module)
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in MODULES:
            origin = getattr(module, '__file__', None)
            if origin is None or origins.get(Path(origin).resolve()) != MODULES[name.split('.')[0]]:
                raise RuntimeError('validator module origin is outside its pinned distribution')
    return {'schema_version': 1, 'python': sys.version, 'interpreter': str(Path(sys.executable).absolute()),
            'interpreter_sha256': hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            'bytecode_policy': 'fresh-private-cache-prefix-with-writes-disabled',
            'packages': packages, 'inputs': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (SCHEMA, CATALOG, Path(__file__).resolve())}}


def validator(current=False):
    prepare_imports()
    from jsonschema import Draft202012Validator, validators, ValidationError
    from referencing import Registry

    def dimensions(validator, names, instance, schema):
        if not isinstance(instance, list):
            return
        seen = set()
        for row in instance:
            if not isinstance(row, dict) or any(type(row.get(k)) not in (int, Decimal) for k in names):
                continue  # Ordinary schema validation reports malformed rows.
            key = tuple(row[k] for k in names)
            if key in seen:
                yield ValidationError('duplicate metric dimensions')
                return
            seen.add(key)

    schema = load_document(SCHEMA.read_bytes())
    # Only this trusted local artifact is accepted, never an input's $schema or
    # $ref. Registry has no retrieval function; a URI can never cause network I/O.
    Draft202012Validator.check_schema(schema)
    if current:
        schema['$ref'] = '#/$defs/currentProducer'
    checker = Draft202012Validator.TYPE_CHECKER.redefine(
        'integer', lambda _, value: type(value) is int or
        (type(value) is Decimal and value.is_finite() and value == value.to_integral_value()))
    cls = validators.extend(Draft202012Validator, {'x-unique-dimensions': dimensions}, type_checker=checker)
    return cls(schema, registry=Registry())


class SnapshotValidator:
    def __init__(self, current=False):
        self.validator = validator(current)
        self.events = {e['code']: e['name'] for e in load_document(CATALOG.read_bytes())['events']}

    def validate(self, document):
        error = next(self.validator.iter_errors(document), None)
        if error is not None:
            # Error messages and instance paths may contain sensitive strings.
            # The keyword alone is safe and adequate for automated classification.
            raise InvalidSnapshot('snapshot schema violation: ' + str(error.validator))
        for record in document['recent_records']:
            expected = self.events.get(record['event_code'])
            if expected is not None and record['event_name'] != expected:
                raise InvalidSnapshot('known event code/name mismatch')
        return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path, nargs='?')
    parser.add_argument('--current-producer', action='store_true')
    parser.add_argument('--identity', action='store_true')
    args = parser.parse_args()
    try:
        dependencies = identity()
    except (OSError, RuntimeError, importlib.metadata.PackageNotFoundError):
        print('Diagnostic validator dependency unavailable or mismatched', file=sys.stderr)
        return 125
    if args.identity:
        print(json.dumps(dependencies, sort_keys=True))
        return 0
    if args.snapshot is None:
        parser.error('snapshot is required without --identity')
    try:
        with args.snapshot.open('rb') as source:
            document = load_document(source.read(MAX_BYTES + 1))
        SnapshotValidator(args.current_producer).validate(document)
    except (OSError, InvalidSnapshot) as error:
        print(str(error) if isinstance(error, InvalidSnapshot) else 'Cannot read snapshot', file=sys.stderr)
        return 1
    print('CopyDiagnostics snapshot valid (' + ('current producer' if args.current_producer else 'schema-version-1 consumer') + ')')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
