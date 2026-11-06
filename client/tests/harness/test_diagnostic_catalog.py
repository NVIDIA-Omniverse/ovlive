# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent consumer catalog: ABI coverage, safe native fixture generation and emitter coverage."""
import ast
import copy
import json
from pathlib import Path
import re
import unittest
from tools.live.diagnostic_catalog import load, render

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / 'tests/contracts/diagnostic-catalog.json'
NATIVE = 'tests/native/runtime/'
REFERENCE = 'tests/reference_service/'
# Each catalog event's observing tests: a test, or the helper a test calls, that
# causes the event through its production emitter and asserts the emitted
# record's code or name and its key fields. A test that calls the registry's
# emit directly, or counts the event only in metrics, is not one. The registry
# itself emits records_suppressed, so its virtual-clock test observes that
# event's emitter. Multi-variant events list one case per emitter variant
# that a production path reaches; the status variant of stream_disconnected and
# stream_resumed covers an endpoint entering RECONNECTING, which no endpoint
# status does, so no test can cause it.
EMITTER_COVERAGE = {
    'endpoint_registered': [NATIVE + 'diagnostics_api.cpp::endpointAndCredentialRecords'],
    'endpoint_closed': [NATIVE + 'diagnostics_api.cpp::endpointAndCredentialRecords',
                        REFERENCE + 'test_native.py::test_public_auth_wait_refresh_failure_and_cancellation'],
    'rpc_completed': [NATIVE + 'layer_write_lifecycle.cpp::testLayerWriteLifecycle',
                      REFERENCE + 'test_diagnostics_sample.py::test_json_records_and_snapshot_count_read_only_list'],
    'rpc_failed': [NATIVE + 'layer_write_lifecycle.cpp::testLayerWriteLifecycle',
                   REFERENCE + 'test_graph_transport.py::test_http2_cancel_reset_resumes_read_on_same_connection',
                   REFERENCE + 'test_session.py::test_session_keepalive_timeout_rejoins_and_peers_see_one_left'],
    'auth_challenge': [REFERENCE + 'test_native.py::test_public_auth_wait_refresh_failure_and_cancellation'],
    'auth_replaced': [NATIVE + 'diagnostics_api.cpp::endpointAndCredentialRecords',
                      REFERENCE + 'test_native.py::test_public_auth_wait_refresh_failure_and_cancellation'],
    'auth_failed': [REFERENCE + 'test_native.py::test_public_auth_wait_refresh_failure_and_cancellation'],
    'stream_disconnected': [REFERENCE + 'test_graph_transport.py::test_http2_cancel_reset_resumes_read_on_same_connection',
                            REFERENCE + 'test_session.py::check_reconnect'],
    'stream_resumed': [REFERENCE + 'test_graph_transport.py::test_http2_cancel_reset_resumes_read_on_same_connection',
                       REFERENCE + 'test_session.py::check_reconnect'],
    'layer_ready': [REFERENCE + 'test_graph_transport.py::test_http2_cancel_reset_resumes_read_on_same_connection'],
    'object_replaced': [REFERENCE + 'test_scene_creation.py::test_creation_replacement_before_attach_keeps_confirmed_object'],
    'recovery_gate_closed': [REFERENCE + 'test_graph_transport.py::test_http2_cancel_reset_resumes_read_on_same_connection'],
    'recovery_gate_opened': [REFERENCE + 'test_graph_transport.py::test_http2_cancel_reset_resumes_read_on_same_connection'],
    'outcome_unknown': [REFERENCE + 'test_graph_transport.py::test_http2_reset_after_commit_retains_unknown_write_without_replay',
                        REFERENCE + 'test_scene_creation.py::test_creation_unknown_diagnostic_correlates_with_retained_seed'],
    'backpressure_entered': [NATIVE + 'diagnostics_api.cpp::backpressureExitRecordsAndExport'],
    'backpressure_exited': [NATIVE + 'diagnostics_api.cpp::backpressureExitRecordsAndExport'],
    'save_completed': [REFERENCE + 'test_scene.py::test_save_diagnostics_exclude_dirty_queries_and_keep_the_failed_layer'],
    'drain_completed': [NATIVE + 'diagnostics_api.cpp::drainRecordsFromThePublicWait'],
    'drain_timeout': [NATIVE + 'diagnostics_api.cpp::drainRecordsFromThePublicWait'],
    'quiescence_failed': [NATIVE + 'runtime_foundation.cpp::shutdownTimeoutRetainsOwnership'],
    'retry_scheduled': [
        REFERENCE + 'test_native.py::test_diagnostics_counts_real_transport_attempts_and_exports_json',
        REFERENCE + 'test_graph_transport.py::test_http2_cancel_reset_resumes_read_on_same_connection',
        REFERENCE + 'test_session.py::check_reconnect',
        REFERENCE + 'test_graph_transport.py::test_unavailable_refusal_that_applied_nothing_is_reissued_unchanged_after_backoff',
        REFERENCE + 'test_scene_creation.py::test_creation_refused_with_nothing_applied_is_reissued_after_the_retry_delay',
        REFERENCE + 'test_graph_transport.py::test_unavailable_without_the_service_error_info_resolves_through_the_chain',
        REFERENCE + 'test_scene.py::held_then_issued'],
    'records_suppressed': [NATIVE + 'diagnostics.cpp::deterministicWarningSuppressionAndSummaries'],
    'message_accepted': [REFERENCE + 'test_session.py::check_message_records'],
    'message_write_completed': [REFERENCE + 'test_session.py::check_message_records'],
    'message_received': [REFERENCE + 'test_session.py::test_session_two_connections_and_copied_fifo_messages',
                         REFERENCE + 'test_session.py::check_message_records'],
    'message_discarded': [REFERENCE + 'test_session.py::check_message_records'],
    'message_rejected': [REFERENCE + 'test_session.py::test_session_pending_queue_limit_preserves_accepted_fifo'],
    'callback_failed': [NATIVE + 'diagnostics_api.cpp::callbackFailuresPreserveOutcomes',
                        REFERENCE + 'test_session.py::test_session_callback_failure_preserves_receipts_and_correlates_diagnostics'],
    'flush_unsupported': [REFERENCE + 'test_scene.py::assert_flush_unsupported_record'],
    'host_edits_discarded': [REFERENCE + 'test_scene.py::assert_host_edits_discarded'],
    'read_paused': [REFERENCE + 'test_native.py::actor_read_pause'],
    'read_resumed': [REFERENCE + 'test_native.py::actor_read_pause'],
    'apply_lag_entered': [REFERENCE + 'test_scene.py::test_production_sdf_status_reports_read_versus_applied_lag'],
    'apply_lag_exited': [REFERENCE + 'test_scene.py::test_production_sdf_status_reports_read_versus_applied_lag'],
    'section_replaced': [REFERENCE + 'test_graph_transport.py::assert_section_replaced'],
    'call_budget_exhausted': [REFERENCE + 'test_native.py::test_endpoint_call_budget_bounds_open_calls'],
    'cleanup_timeout': [NATIVE + 'diagnostics_api.cpp::cleanupTimeoutAndBusyResults'],
}


def called_from_test(path, name):
    """Whether a test_* function in the Python file `path` calls `name`, directly or as a method."""
    for node in ast.walk(ast.parse((ROOT / path).read_text(encoding='utf-8'))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_'):
            for call in ast.walk(node):
                if isinstance(call, ast.Call):
                    callee = call.func
                    if (callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, 'id', None)) == name:
                        return True
    return False


def case_source(path, name):
    """The source of the test or helper `name` in `path`, or None."""
    text = (ROOT / path).read_text(encoding='utf-8')
    if path.endswith('.py'):
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return ast.get_source_segment(text, node)
        return None
    match = re.search(r'^(?:static\s+)?void\s+' + re.escape(name) + r'\s*\([^)]*\)\s*\{', text, re.M)
    if not match:
        return None
    # Match braces outside string and character literals and comments.
    depth, index = 0, match.end() - 1
    while index < len(text):
        if text.startswith('//', index):
            index = text.find('\n', index)
            continue
        if text.startswith('/*', index):
            index = text.find('*/', index) + 2
            continue
        char = text[index]
        if char in '"\'':
            index += 1
            while text[index] != char:
                index += 2 if text[index] == '\\' else 1
        elif char in '{}':
            depth += 1 if char == '{' else -1
            if depth == 0:
                return text[match.start():index + 1]
        index += 1
    return None


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
            lambda d: d['cleanup_calls'][0].update(value=0),
            lambda d: d['cleanup_calls'].pop(),
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
        self.assertIn('static_assert(kDiagnosticEventCount == 39);', output)
        self.assertIn('assert(std::string_view(DiagnosticsRegistry::eventName(23)) == "invalid");', output)

    def test_generated_checks_bind_compiled_registry_and_rpc_contract(self):
        catalog = load(CATALOG.read_bytes())
        output = render(catalog)
        self.assertIn('DiagnosticsRegistry::eventName(OMNI_USD_LIVE_RECORD_RPC_FAILED)', output)
        self.assertIn('RuntimeOptions::copy(nullptr)', output)
        self.assertIn('kMetricDurationBoundsMs[7] == 5000', output)
        self.assertIn('static_assert(static_cast<uint32_t>(CleanupCall::LeaveSession) == 4);', output)
        self.assertIn('catalog_rpc_cases', output)
        self.assertIn('catalog_rpc_fields', output)
        self.assertNotIn('severity_rule', output)  # Prose is never evaluated as code.


class EmitterCoverageTests(unittest.TestCase):
    def test_every_catalog_event_names_an_observing_test(self):
        events = {e['name']: e['symbol'] for e in load(CATALOG.read_bytes())['events']}
        self.assertEqual(sorted(EMITTER_COVERAGE), sorted(events))
        for name, cases in EMITTER_COVERAGE.items():
            self.assertTrue(cases, name)
            for case in cases:
                with self.subTest(event=name, case=case):
                    path, function = case.split('::')
                    source = case_source(path, function)
                    self.assertIsNotNone(source, 'no test or helper ' + case)
                    # A Python helper counts only when a test in its file calls it.
                    if path.endswith('.py') and not function.startswith('test_'):
                        self.assertTrue(called_from_test(path, function), case + ' is called by no test_* function')
                    # The case selects the record by its name or its ABI constant.
                    self.assertTrue(re.search(r"""['"]%s['"]|\\"%s\\"|\b%s\b""" % (name, name, events[name]), source),
                                    case + ' does not name ' + name)

    def test_case_source_extracts_whole_definitions_only(self):
        self.assertIsNone(case_source(NATIVE + 'diagnostics_api.cpp', 'noSuchCase'))
        body = case_source(NATIVE + 'diagnostics_api.cpp', 'drainRecordsFromThePublicWait')
        self.assertTrue(body.startswith('void drainRecordsFromThePublicWait()') and body.endswith('}'))
        self.assertNotIn('void subjectMetricsTrackAttachedLatchedStates', body)
        self.assertIn('def check_reconnect', case_source(REFERENCE + 'test_session.py', 'check_reconnect'))
        # A helper counts only through a test in its file that calls it.
        self.assertTrue(called_from_test(REFERENCE + 'test_session.py', 'check_message_records'))
        self.assertFalse(called_from_test(REFERENCE + 'test_session.py', 'no_such_helper'))
        self.assertFalse(called_from_test(REFERENCE + 'test_native.py', 'check_message_records'))
        # Braces inside literals and comments do not end a definition.
        body = case_source(NATIVE + 'diagnostics_api.cpp', 'backpressureExitRecordsAndExport')
        self.assertTrue(body.endswith('}') and 'void applicationMetricsExportWithoutHistory' not in body)


if __name__ == '__main__':
    unittest.main()
