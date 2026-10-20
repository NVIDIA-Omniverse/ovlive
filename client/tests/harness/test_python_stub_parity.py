# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shipped stub names exactly the public runtime surface, without a build.

The comparison is static: it reads top-level definitions, assignments and the
relative `._host` re-export, plus class-body members and `self.<name>`
attributes stored in runtime `__init__` methods. Names created dynamically or
conditionally are not seen. Any other relative import at module level fails the
check, so a new re-export source cannot slip past it; `from . import _live`
imports the private extension and exports nothing.

Every dunder except `__init__` is public surface. An exported class's stub lists
every dunder its runtime class body defines or assigns, including `__repr__`,
`__eq__`, `__hash__` and `__slots__`, and no other; dunders the class inherits
from `object` without redefining them are not compared.
"""
import ast
from pathlib import Path
import unittest

PACKAGE = Path(__file__).resolve().parents[2] / 'source/bindings/python/omni/usd/live'


def public(name):
    """Unprefixed names and every dunder except `__init__` are public surface."""
    dunder = len(name) > 4 and name.startswith('__') and name.endswith('__')
    return not name.startswith('_') or (dunder and name != '__init__')


def module_names(path):
    names = set()
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom) and node.level:
            if node.level == 1 and node.module == '_host':
                names.update(alias.asname or alias.name for alias in node.names)
            elif not (node.level == 1 and node.module is None and
                      [(alias.name, alias.asname) for alias in node.names] == [('_live', None)]):
                raise AssertionError(f'{path.name}:{node.lineno}: unsupported relative import')
    return {name for name in names if public(name)}


def class_members(path, runtime):
    classes = {}
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if not isinstance(node, ast.ClassDef):
            continue
        members = set()
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                members.add(item.name)
                if runtime and item.name == '__init__':
                    members.update(target.attr for target in ast.walk(item)
                                   if isinstance(target, ast.Attribute) and isinstance(target.ctx, ast.Store)
                                   and isinstance(target.value, ast.Name) and target.value.id == 'self')
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                members.add(item.target.id)
            elif isinstance(item, ast.Assign):
                members.update(t.id for t in item.targets if isinstance(t, ast.Name))
        classes[node.name] = {name for name in members if public(name) and name != '__init__'}
    return classes


class StubParityTests(unittest.TestCase):
    def test_module_names_match(self):
        runtime = module_names(PACKAGE / '__init__.py')
        stub = module_names(PACKAGE / '__init__.pyi')
        self.assertEqual(sorted(runtime - stub), [], 'public runtime names missing from the stub')
        self.assertEqual(sorted(stub - runtime), [], 'stub names without a runtime definition')

    def test_class_members_match(self):
        runtime = class_members(PACKAGE / '__init__.py', True)
        runtime.update(class_members(PACKAGE / '_host.py', True))
        stub = class_members(PACKAGE / '__init__.pyi', False)
        exported = module_names(PACKAGE / '__init__.py')
        for name in sorted(exported & set(runtime)):
            with self.subTest(cls=name):
                self.assertIn(name, stub)
                self.assertEqual(sorted(runtime[name] ^ stub[name]), [], f'{name} members differ')


if __name__ == '__main__':
    unittest.main()
