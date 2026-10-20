# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-case test certificate authorities from the pinned OpenSSL (LIVE_TEST_OPENSSL).

Keys are synthetic and live in the caller's temporary directory.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import ssl
import subprocess
import tempfile

NAME = 'live.test'


@dataclass(frozen=True)
class TestPki:
    ca: Path
    certificate: Path
    key: Path

    def server_context(self):
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(self.certificate, self.key)
        context.set_alpn_protocols(['h2'])
        return context

    def environment(self):
        """Native driver TLS settings (tests/native/support/test_tls.h)."""
        return dict(LIVE_TEST_TLS_CA=str(self.ca), LIVE_TEST_TLS_NAME=NAME)


def generate(directory, label='ca'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / 'openssl.cnf'
    config.write_text('')

    def run(*arguments):
        subprocess.run([os.environ['LIVE_TEST_OPENSSL'], *map(str, arguments)], check=True, capture_output=True, timeout=30)
    # Both certificates start one day in the past: a wall-clock step back
    # (such as a WSL2 time sync) between issue and handshake would otherwise
    # make them not yet valid. `-days` counts from the time of issue.
    not_before = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y%m%d%H%M%SZ')
    ca, ca_key = directory / f'{label}.pem', directory / f'{label}.key'
    run('req', '-config', config, '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1', '-not_before', not_before,
        '-keyout', ca_key, '-out', ca, '-subj', f'/CN={label} test CA', '-addext', 'basicConstraints=critical,CA:TRUE',
        '-addext', 'keyUsage=critical,keyCertSign')
    key, request = directory / f'{label}-server.key', directory / f'{label}-server.csr'
    certificate, extensions = directory / f'{label}-server.pem', directory / f'{label}-server.ext'
    run('req', '-config', config, '-newkey', 'rsa:2048', '-nodes', '-keyout', key, '-out', request, '-subj', f'/CN={NAME}')
    extensions.write_text(f'subjectAltName=DNS:{NAME}\nextendedKeyUsage=serverAuth\n')
    run('x509', '-req', '-in', request, '-CA', ca, '-CAkey', ca_key, '-CAcreateserial', '-days', '1',
        '-not_before', not_before, '-extfile', extensions, '-out', certificate)
    return TestPki(ca, certificate, key)


async def generate_async(directory, label='ca'):
    # Off the event loop, so service workers are not stalled by key generation.
    return await asyncio.to_thread(generate, directory, label)


async def case_pki(test, label='ca'):
    """The test case's PKI for label, generated once per case in a directory the case's cleanup removes."""
    if not hasattr(test, 'pkis'):
        directory = tempfile.TemporaryDirectory()
        test.addCleanup(directory.cleanup)
        test.pki_directory, test.pkis = Path(directory.name), {}
    if label not in test.pkis:
        test.pkis[label] = await generate_async(test.pki_directory, label)
    return test.pkis[label]
