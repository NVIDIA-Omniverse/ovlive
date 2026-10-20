#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render explicit Live Services package pins; never fetch or discover credentials."""
import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == content:
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def generate(root: Path, target_abi, host_abi):
    rendered = {}
    for role, value in [('target', target_abi), ('host', host_abi)]:
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', value):
            raise ValueError('the ' + role + ' ABI must explicitly name the pinned package ABI')
    for role, abi in [('target', target_abi), ('host', host_abi)]:
        content = (root / 'deps' / f'live-services-{role}.packman.xml').read_text()
        token = '${live_services_' + role + '_platform_abi}'
        if token not in content:
            raise ValueError('missing explicit ABI placeholder in ' + role + ' dependency template')
        content = content.replace(token, abi)
        ET.fromstring(content)
        rendered[root / 'deps' / f'live-services-{role}.generated.packman.xml'] = content
    # All validation precedes mutation. Provisioning consumes only these rendered
    # files; no client checkout, discovery provider or implicit ABI guess exists.
    for path, content in rendered.items():
        _write(path, content)
    _write(root / 'deps/live-services-build.json', json.dumps({
        'schema_version': 1, 'enabled': True,
        'target_platform_abi': target_abi,
        'host_platform_abi': host_abi,
    }, indent=2, sort_keys=True) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Render the Live Services dependency manifests for explicit ABIs.')
    parser.add_argument('--target-abi', required=True, help='ABI token of the target gRPC closure packages')
    parser.add_argument('--host-abi', required=True, help='ABI token of the host protoc/grpc_cpp_plugin packages')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2],
                        help='client root holding deps/ (default: this checkout)')
    args = parser.parse_args(argv)
    try:
        generate(args.root, args.target_abi, args.host_abi)
    except (OSError, ValueError, ET.ParseError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    sys.exit(main())
