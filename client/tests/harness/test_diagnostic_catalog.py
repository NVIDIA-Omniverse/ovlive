# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent consumer catalog: ABI coverage and safe native fixture generation."""
import copy
import json
from pathlib import Path
import re
import unittest
from tools.live.diagnostic_catalog import load, render

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / 'tests/contracts/diagnostic-catalog.json'


def enum_blocks(header):
    # Fail closed on unfamiliar enum syntax instead of silently skipping members.
    header = re.sub(r'/\*.*?\*/|//[^\n]*', '', header, flags=re.S)
    blocks = []
    for block in re.findall(r'enum\s*\{(.*?)\};', header, re.S):
        values = {}
        for member in block.split(','):
            if not member.strip():
                continue
            match = re.fullmatch(r'\s*(OMNI_USD_LIVE_[A-Z_0-9]+)\s*=\s*(0|[1-9][0-9]*)\s*', member)
            if not match or match[1] in values:
                raise ValueError('unrecognized or duplicate public enum member: ' + member.strip())
            values[match[1]] = int(match[2])
        blocks.append(values)
    return blocks


class DiagnosticCatalogTests(unittest.TestCase):
    def test_catalog_covers_entire_public_enum_domains(self):
        catalog = load(CATALOG.read_bytes())
        header = (ROOT / 'include/omni/usd/live/Live.h').read_text()
        blocks = enum_blocks(header)
        for domain, rows in catalog['domains'].items():
            with self.subTest(domain=domain):
                expected = {r['symbol']: r['value'] for r in rows}
                actual = [block for block in blocks if rows[0]['symbol'] in block]
                self.assertEqual(actual, [expected])
        events = {e['symbol']: e['code'] for e in catalog['events']}
        self.assertEqual([b for b in blocks if 'OMNI_USD_LIVE_RECORD_ENDPOINT_REGISTERED' in b], [events])

    def test_enum_coverage_never_silently_omits_new_members(self):
        self.assertEqual(enum_blocks('enum { OMNI_USD_LIVE_RESULT_V2 = 23 };'),
                         [{'OMNI_USD_LIVE_RESULT_V2': 23}])
        for member in ('OMNI_USD_LIVE_NEW', 'OMNI_USD_LIVE_ALIAS = OMNI_USD_LIVE_OK',
                       'OMNI_USD_LIVE_NEW = 0x20', 'OMNI_USD_LIVE_OK = 1'):
            with self.subTest(member=member), self.assertRaises(ValueError):
                enum_blocks('enum { OMNI_USD_LIVE_OK = 0, ' + member + ' };')

    def test_sparse_result_fixture_uses_explicit_codes(self):
        catalog = load(CATALOG.read_bytes())
        catalog['domains']['results'][-1]['value'] = 50
        checked = load(json.dumps(catalog))
        output = render(checked)
        self.assertIn('static_assert(OMNI_USD_LIVE_IO_ERROR == 50)', output)
        self.assertIn('{OMNI_USD_LIVE_IO_ERROR, OMNI_USD_LIVE_SEVERITY_ERROR}', output)

    def test_generator_rejects_ambiguous_and_unsafe_contracts(self):
        original = load(CATALOG.read_bytes())
        mutations = [
            lambda d: d['events'][0].update(code=True),
            lambda d: d['events'][1].update(code=d['events'][0]['code']),
            lambda d: d['events'][0].update(name='bad"; code(); //'),
            lambda d: d['fields'][0].update(key='bad); code(); //'),
            lambda d: d['events'][0].update(variants=['missing']),
            lambda d: d['events'][19].update(emission='reserved'),
            lambda d: d['shutdown_phases'][0].update(value=9),
            lambda d: d['operation_auth_wait_kinds'][0].update(value=3),
            lambda d: d['events'][3]['result_severities'].pop('OMNI_USD_LIVE_TIMEOUT'),
            lambda d: d['variants']['rpc']['fields'].append('result'),
            lambda d: d['metric_contract'].update(duration_bounds_ms=[1, 1]),
            lambda d: d['defaults'].update(max_diagnostic_records=True),
            lambda d: d['host_actions'].pop(),
            lambda d: d.update(retired_event_codes=[]),
            lambda d: d.update(retired_event_codes=[d['events'][0]['code']]),
            lambda d: d.update(retired_event_codes=[23, 23]),
            lambda d: d.update(retired_event_codes=[0, 23]),
            lambda d: d['variants'].update(unused=copy.deepcopy(d['variants']['rpc'])),
            lambda d: d.pop('retired_domain_values'),
            lambda d: d['retired_domain_values'].update(events=[23]),
            lambda d: d['retired_domain_values'].update(results=[0]),
            lambda d: d['retired_domain_values'].update(states=[6, 6]),
            lambda d: d['retired_domain_values'].update(operations=[True]),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutations.index(mutation)):
                candidate = copy.deepcopy(original)
                mutation(candidate)
                with self.assertRaises(ValueError):
                    load(json.dumps(candidate))
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            load('{"catalog_version":1,"catalog_version":1}')

    def test_retired_event_codes_keep_the_numbering_contiguous(self):
        data = json.loads(CATALOG.read_text())
        codes = {event['code'] for event in data['events']}
        retired = set(data['retired_event_codes'])
        self.assertEqual(retired, {23})
        self.assertFalse(codes & retired)
        self.assertEqual(codes | retired, set(range(1, max(codes | retired) + 1)))

    def test_retired_domain_values_name_no_public_member(self):
        catalog = load(CATALOG.read_bytes())
        self.assertEqual(catalog['retired_domain_values'], dict(operations=[15], results=[17], states=[6]))
        blocks = enum_blocks((ROOT / 'include/omni/usd/live/Live.h').read_text())
        for domain, retired in catalog['retired_domain_values'].items():
            with self.subTest(domain=domain):
                block, = [b for b in blocks if catalog['domains'][domain][0]['symbol'] in b]
                self.assertFalse(set(retired) & set(block.values()))
        output = render(catalog)
        self.assertIn('static_assert(!definedResult(17u));', output)
        self.assertIn('static_assert(!definedState(6u));', output)

    def test_generated_checks_keep_retired_event_slots_reserved(self):
        output = render(load(CATALOG.read_bytes()))
        self.assertIn('static_assert(kDiagnosticEventCount == 36);', output)
        self.assertIn('assert(std::string_view(DiagnosticsRegistry::eventName(23)) == "invalid");', output)

    def test_generated_checks_bind_compiled_registry_and_rpc_contract(self):
        catalog = load(CATALOG.read_bytes())
        output = render(catalog)
        self.assertIn('DiagnosticsRegistry::eventName(OMNI_USD_LIVE_RECORD_RPC_FAILED)', output)
        self.assertIn('RuntimeOptions::copy(nullptr)', output)
        self.assertIn('kMetricDurationBoundsMs[7] == 5000', output)
        self.assertIn('catalog_rpc_cases', output)
        self.assertIn('catalog_rpc_fields', output)
        self.assertNotIn('severity_rule', output)  # Prose is never evaluated as code.


if __name__ == '__main__':
    unittest.main()
