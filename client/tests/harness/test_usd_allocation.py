# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Crashes, partial evidence and incoherent controls cannot finish an allocation sweep."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.live.usd_allocation import records, terminal, unique_selections


class AllocationDiagnosticTests(unittest.TestCase):
    def entry(self):
        return dict(process=dict(outcome='passed'), records=[dict(phase='armed'),
            dict(phase='returned', injected=False, caught_bad_alloc=False, returned_success=True),
            dict(phase='observed', baseline_unchanged=True, retained_coherent=True,
                 stage_namespace_coherent=True, stage_values_coherent=True, expected_effect=True, sdk_diagnostics=False)])

    def test_only_clean_uninjected_success_finishes_sweep(self):
        value = self.entry()
        self.assertTrue(terminal(value))
        for outcome in ('failed', 'timeout', 'signaled'):
            changed = copy.deepcopy(value)
            changed['process']['outcome'] = outcome
            self.assertFalse(terminal(changed))
        for key in ('injected', 'caught_bad_alloc'):
            changed = copy.deepcopy(value)
            changed['records'][1][key] = True
            self.assertFalse(terminal(changed))
        for key in ('baseline_unchanged', 'retained_coherent', 'stage_namespace_coherent', 'stage_values_coherent', 'expected_effect'):
            changed = copy.deepcopy(value)
            changed['records'][2][key] = False
            self.assertFalse(terminal(changed))
        for count in (1, 2):
            changed = copy.deepcopy(value)
            changed['records'] = changed['records'][:count]
            self.assertFalse(terminal(changed))

    def test_duplicate_sweeps_cannot_overwrite_prior_logs(self):
        self.assertIsNone(unique_selections(None))
        self.assertEqual(unique_selections(['stock', 'projection']), ['stock', 'projection'])
        for values in (['stock', 'stock'], ['move', 'move']):
            with self.assertRaises(ValueError):
                unique_selections(values)

    def test_missing_armed_record_and_redirected_sdk_are_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sdk = root / 'sdk'
            sdk.mkdir()
            library = sdk / 'libusd_sdf.so'
            library.write_bytes(b'synthetic SDK identity')
            armed = dict(phase='armed', backing='stock', operation='move', index=3,
                         usd_version=2511, sdf_library=str(library))
            def encode(value):
                return json.dumps(value, separators=(',', ':')) + '\n'
            self.assertEqual(records(encode(armed), 'stock', 'move', 3, sdk)['records'], [armed])
            for output in ('', encode(dict(phase='returned')), encode(armed) + encode(armed)):
                with self.assertRaises(ValueError):
                    records(output, 'stock', 'move', 3, sdk)
            library.unlink()
            outside = root / 'elsewhere.so'
            outside.write_bytes(b'different SDK')
            library.symlink_to(outside)
            with self.assertRaises(ValueError):
                records(encode(armed), 'stock', 'move', 3, sdk)


if __name__ == '__main__':
    unittest.main()
