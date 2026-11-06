# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The frozen work budgets cover every storage benchmark workload and every kind of case."""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
BUDGETS = ROOT / 'tests/native/scene/work_budgets.cpp'


class WorkBudgetTests(unittest.TestCase):
    def test_storage_cases_are_the_benchmark_workloads_and_each_is_frozen(self):
        source = BUDGETS.read_text(encoding='utf-8')
        cases = re.findall(r'\{"(\w+)", "(\w+)", (\d+), (\d+), (\d+), (\d+)\}', source)
        workloads = json.loads((ROOT / 'tests/benchmarks/scene-workloads.json').read_text(encoding='utf-8'))['cases']
        self.assertEqual(cases, [(case['id'], case['kind'], str(case['specs']), str(case['operations']),
                                  str(case['array_elements']), str(case['readers'])) for case in workloads])
        frozen = re.findall(r'\{"([\w.]+)", (\d+), (\d+), (\d+)\}', source)
        ids = [entry[0] for entry in frozen]
        self.assertEqual(len(ids), len(set(ids)))
        for case in workloads:
            self.assertIn('storage.' + case['id'], ids)
        for prefix in ('turn.', 'publication.', 'flush.capture', 'read.'):
            self.assertTrue(any(name.startswith(prefix) for name in ids), prefix)


if __name__ == '__main__':
    unittest.main()
