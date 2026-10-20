#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the consumer-owned diagnostic catalog and emit native contract checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', value), 'invalid identifier')
    return value


def number(value):
    require(type(value) is int and 0 <= value <= (1 << 64) - 1, 'invalid unsigned value')
    return value


def load(content):
    data = json.loads(content, object_pairs_hook=unique_object)
    require(type(data) is dict and type(data.get('catalog_version')) is int and data['catalog_version'] == 1,
            'unsupported catalog version')
    require(type(data.get('diagnostic_schema_version')) is int and data['diagnostic_schema_version'] == 1,
            'unsupported diagnostic schema')
    for key in ('purpose', 'compatibility', 'delivery', 'field_presence', 'default_policy', 'limits'):
        require(isinstance(data.get(key), str) and bool(data[key]), 'missing catalog policy')
    fields = data['fields']; events = data['events']; domains = data['domains']
    require(isinstance(fields, list) and 1 <= len(fields) <= 64, 'invalid fields')
    names = set(); keys = set()
    for field in fields:
        name = identifier(field['name']); require(name not in names, 'duplicate field'); names.add(name)
        require(bool(field['meaning']), 'missing field meaning')
        if field['key'] is None:
            require(name == 'canonical_resource' and field['kind'] == 'STRING', 'invalid resource field')
        else:
            key = identifier(field['key']); require(key not in keys, 'duplicate internal field'); keys.add(key)
            require(field['kind'] == 'UINT64', 'invalid numeric field kind')
    symbols = set()
    require(set(domains) == {'operations', 'results', 'states', 'recovery_actions', 'auth_reasons', 'severities'},
            'missing enum domain')
    # A retired value keeps its number reserved: it names no member of its domain.
    retired_values = data.get('retired_domain_values')
    require(isinstance(retired_values, dict) and set(retired_values) <= set(domains), 'invalid retired domain values')
    for domain, rows in domains.items():
        values = set()
        require(isinstance(rows, list) and bool(rows), 'empty enum domain')
        for row in rows:
            symbol = identifier(row['symbol']); value = number(row['value'])
            require(symbol.startswith('OMNI_USD_LIVE_') and symbol not in symbols and value not in values,
                    'duplicate/invalid enum member')
            symbols.add(symbol); values.add(value)
        retired = retired_values.get(domain, [])
        require(isinstance(retired, list) and len(set(retired)) == len(retired), 'invalid retired domain values')
        require(not {number(value) for value in retired} & values, 'retired domain value is defined')
    severities = {r['symbol'].removeprefix('OMNI_USD_LIVE_SEVERITY_'): r['value'] for r in domains['severities']}
    for variant in data['variants'].values():
        used = variant['fields'] + variant['optional']
        require(len(set(used)) == len(used) and set(used) <= names and bool(variant['meaning']), 'invalid variant')
    require(isinstance(events, list) and 1 <= len(events) <= 256, 'invalid events')
    codes = set(); event_names = set()
    for event in events:
        code = number(event['code']); name = identifier(event['name']); symbol = identifier(event['symbol'])
        require(code > 0 and code not in codes and name not in event_names and symbol not in symbols,
                'duplicate event')
        require(symbol == 'OMNI_USD_LIVE_RECORD_' + name.upper(), 'event symbol/name mismatch')
        codes.add(code); event_names.add(name); symbols.add(symbol)
        require(event['default_severity'] in severities, 'unknown severity')
        require(event['emission'] in {'implemented', 'reserved'} and bool(event['meaning']), 'invalid emission')
        require(bool(event['variants']) == (event['emission'] == 'implemented'), 'reserved/implemented variant mismatch')
        require(set(event['variants']) <= set(data['variants']), 'unknown variant')
        if 'result_severities' in event:
            require(set(event['result_severities']) == {r['symbol'] for r in domains['results']}, 'incomplete result severities')
            require(set(event['result_severities'].values()) <= set(severities), 'unknown result severity')
    retired = data.get('retired_event_codes', [])
    require(isinstance(retired, list) and len(set(retired)) == len(retired), 'invalid retired event codes')
    retired = {number(code) for code in retired}
    require(all(code > 0 for code in retired) and not codes & retired, 'retired event code is defined')
    require(codes | retired == set(range(1, max(codes | retired) + 1)),
            'defined and retired event codes must cover the ABI without gaps')
    require({v for event in events for v in event['variants']} == set(data['variants']), 'unused variant')
    require({r['symbol'] for r in data['host_actions']} == {r['symbol'] for r in domains['recovery_actions']},
            'host action coverage mismatch')
    for action in data['host_actions']:
        require(bool(action['guidance']), 'missing host guidance')
    for name, value in data['defaults'].items():
        identifier(name); require(number(value) > 0, 'zero default')
    phases = data['shutdown_phases']
    require({number(r['value']) for r in phases} == set(range(9)) and
            len({identifier(r['name']) for r in phases}) == 9 and len(phases) == 9, 'invalid shutdown phases')
    kinds = data['operation_auth_wait_kinds']
    require(len(kinds) == 3 and {number(r['value']) for r in kinds} == set(range(3)) and
            len({identifier(r['name']) for r in kinds}) == 3 and all(r['meaning'] for r in kinds),
            'invalid operation auth wait kinds')
    bounds = data['metric_contract']['duration_bounds_ms']
    callbacks = data['callback_kinds']
    require(len(callbacks) == 4 and {number(r['value']) for r in callbacks} == set(range(4)) and
            len({identifier(r['name']) for r in callbacks}) == 4, 'invalid callback kinds')
    require(all(type(v) is int and v > 0 for v in bounds) and bounds == sorted(set(bounds)), 'invalid duration bounds')
    return data


def render(data):
    lines = ['// Generated test contract from diagnostic-catalog.json; not runtime implementation.',
             'void diagnosticCatalogContract()', '{',
             f'    static_assert(OMNI_USD_LIVE_DIAGNOSTIC_SCHEMA_VERSION == {data["diagnostic_schema_version"]});']
    for kind in data['operation_auth_wait_kinds']:
        lines.append(f"    static_assert(static_cast<uint32_t>(AuthWaitKind::{kind['name']}) == {kind['value']});")
    for kind in data['callback_kinds']:
        lines.append(f"    static_assert(static_cast<uint32_t>(CallbackKind::{kind['name']}) == {kind['value']});")
    for phase in data['shutdown_phases']:
        lines.append(f"    static_assert(static_cast<uint32_t>(ShutdownPhase::{phase['name']}) == {phase['value']});")
    for rows in data['domains'].values():
        for row in rows:
            lines.append(f'    static_assert({row["symbol"]} == {row["value"]});')
    for event in data['events']:
        symbol = event['symbol']; name = event['name']; severity = event['default_severity']
        lines += [f'    static_assert({symbol} == {event["code"]});',
                  f'    assert(std::string_view(DiagnosticsRegistry::eventName({symbol})) == "{name}");',
                  f'    assert(DiagnosticsRegistry::eventSeverity({symbol}) == OMNI_USD_LIVE_SEVERITY_{severity});']
    codes = {event['code'] for event in data['events']} | set(data.get('retired_event_codes', []))
    lines.append(f'    static_assert(kDiagnosticEventCount == {max(codes) + 1});')
    for code in sorted(data.get('retired_event_codes', [])):
        lines.append(f'    assert(std::string_view(DiagnosticsRegistry::eventName({code})) == "invalid");')
    for value in data['retired_domain_values'].get('results', []):
        lines.append(f'    static_assert(!definedResult({value}u));')
    for value in data['retired_domain_values'].get('states', []):
        lines.append(f'    static_assert(!definedState({value}u));')
    numeric_fields = [f for f in data['fields'] if f['key'] is not None]
    for field in numeric_fields:
        lines.append(f'    assert(std::string_view(DiagnosticsRegistry::fieldName(DiagnosticKey::{field["key"]})) == "{field["name"]}");')
    lines.append(f'    static_assert(static_cast<size_t>(DiagnosticKey::CountKeys) == {len(numeric_fields)});')
    for index, value in enumerate(data['metric_contract']['duration_bounds_ms']):
        lines.append(f'    static_assert(kMetricDurationBoundsMs[{index}] == {value});')
    lines.append(f"    static_assert(kMetricDurationBoundsMs.size() == {len(data['metric_contract']['duration_bounds_ms'])});")
    lines.append('    const auto normalized = RuntimeOptions::copy(nullptr);')
    for name, value in data['defaults'].items():
        lines.append(f'    assert(normalized.values.{name} == UINT64_C({value}));')
    lines += ['    assert(normalized.values.diagnostic_min_severity == OMNI_USD_LIVE_SEVERITY_INFO);',
              '    assert(normalized.values.diagnostic_allowed_flags == 0);',
              '    assert(normalized.values.flags == 0);',
              '    OmniUsdLiveRuntimeOptions supplied{};',
              '    supplied.struct_size = sizeof(supplied); supplied.abi_version = OMNI_USD_LIVE_ABI_VERSION;',
              '    const auto zeroOptions = RuntimeOptions::copy(&supplied);',
              '    assert(zeroOptions.values.diagnostic_min_severity == OMNI_USD_LIVE_SEVERITY_TRACE);',
              '    assert(zeroOptions.values.max_diagnostic_records == normalized.values.max_diagnostic_records);', '}']
    for event in data['events']:
        if 'result_severities' in event:
            lines.append('struct CatalogRpcCase { uint32_t result, severity; };')
            lines.append(f'constexpr std::array<CatalogRpcCase, {len(data["domains"]["results"])}> catalog_rpc_cases{{{{')
            for result in sorted(data['domains']['results'], key=lambda r: r['value']):
                lines.append('    {' + result['symbol'] + ', OMNI_USD_LIVE_SEVERITY_' + event['result_severities'][result['symbol']] + '},')
            lines.append('}};')
    lines.append(f"constexpr std::array<const char*, {len(data['variants']['rpc']['fields'])}> catalog_rpc_fields{{")
    for field in data['variants']['rpc']['fields']:
        lines.append('    ' + json.dumps(field) + ',')
    lines.append('};')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('catalog', type=Path)
    parser.add_argument('--header', type=Path)
    args = parser.parse_args()
    data = load(args.catalog.read_bytes())
    if args.header:
        args.header.parent.mkdir(parents=True, exist_ok=True)
        args.header.write_text(render(data))
    else:
        print(f'Catalog valid: {len(data["events"])} events, {len(data["fields"])} fields')


if __name__ == '__main__':
    main()
