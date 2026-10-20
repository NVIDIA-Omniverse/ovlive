# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared sanitizer recipe: leak detection on and an owned third-party suppressions file."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.live import sanitizer, test as runner

PRODUCT = re.compile(r'omni::usd::live|live_|omni_usd_live')
# Allocator and libstdc++ frames sit on nearly every allocation stack, so an
# entry naming one would hide every leak.
DENYLIST = re.compile(r'^leak:(operator new|malloc|calloc|realloc|std::|__gnu_cxx|libstdc\+\+|libc\.)')

# A path-keyed entry names a third-party source tree that only third-party
# frames carry: statically linked OpenSSL's crypto/, ssl/ and providers/
# (crypto/rand/ lies under crypto/), USD's pxr/, gRPC's src/core/ and glibc's
# nptl/ and sysdeps/. Product and test trees never qualify.
THIRD_PARTY_TREE = re.compile(r'^/(src/(crypto|ssl|providers)|pxr|src/core|nptl|sysdeps)/')
PRODUCT_TREE = re.compile(r'(^|/)(source|include|tests|tools|plugin|cmake)/')
# A fully anchored entry names one bare third-party C symbol: no namespace, template, parameter list,
# wildcard, Itanium-mangled C++ name (_Z...) or product C API name (omniUsdLive...).
C_SYMBOL = re.compile(r'^\^(?!_Z|omniUsdLive)[A-Za-z_][A-Za-z0-9_]*\$$')


def entry_pattern(line):
    """The pattern of a `kind:pattern` suppression line."""
    return line.split(':', 1)[1]


def suppression_policy_violation(line):
    """Why a path-keyed or fully anchored suppression breaks the third-party policy, or None."""
    pattern = entry_pattern(line)
    if '/' in pattern and not pattern.startswith('^'):
        if PRODUCT_TREE.search(pattern):
            return 'names a product source tree'
        if not THIRD_PARTY_TREE.search(pattern):
            return 'names no third-party source tree'
    if pattern.startswith('^') and pattern.endswith('$') and not C_SYMBOL.match(pattern):
        return 'anchors something other than a bare C symbol'
    return None


def template_match(pattern, name):
    """ThreadSanitizer's suppression match: substring, with ^ and $ anchors and * wildcards."""
    start, end = pattern.startswith('^'), pattern.endswith('$')
    body = pattern[1 if start else 0:len(pattern) - 1 if end else len(pattern)]
    expression = '.*'.join(re.escape(part) for part in body.split('*'))
    return re.search(('^' if start else '') + expression + ('$' if end else ''), name) is not None


class SanitizerRecipeTests(unittest.TestCase):
    def entries(self):
        lines = [line.strip() for line in sanitizer.SUPPRESSIONS.read_text().splitlines()]
        return [(index, line) for index, line in enumerate(lines) if line and not line.startswith('#')], lines

    def test_leak_detection_is_on_with_the_owned_suppressions(self):
        options = sanitizer.runtime_options()
        self.assertEqual(set(options), {'ASAN_OPTIONS', 'LSAN_OPTIONS', 'UBSAN_OPTIONS'})
        self.assertIn('detect_leaks=1', options['ASAN_OPTIONS'].split(':'))
        self.assertIn('fast_unwind_on_malloc=0', options['ASAN_OPTIONS'].split(':'))
        self.assertNotIn('detect_leaks=0', options['ASAN_OPTIONS'])
        self.assertIn('suppressions=' + sanitizer.option_value(sanitizer.SUPPRESSIONS.resolve()),
                      sanitizer.split_options(options['LSAN_OPTIONS']))
        self.assertTrue(sanitizer.SUPPRESSIONS.is_absolute())

    def test_a_path_holding_the_separator_is_one_quoted_option(self):
        drive = 'D:\\a\\ovlive\\tools\\live\\lsan.supp'
        self.assertEqual(sanitizer.option_value(drive), '"' + drive + '"')
        self.assertEqual(sanitizer.option_value('/srv/ovlive/lsan.supp'), '/srv/ovlive/lsan.supp')
        options = f'suppressions={sanitizer.option_value(drive)}:print_suppressions=0'
        self.assertEqual(sanitizer.split_options(options), ['suppressions="' + drive + '"', 'print_suppressions=0'])


    def test_every_suppression_is_a_leak_entry_with_owner_and_reason(self):
        entries, lines = self.entries()
        for index, line in entries:
            with self.subTest(line=line):
                self.assertRegex(line, r'^leak:\S+$')
                self.assertGreater(index, 0)
                comment = lines[index - 1]
                self.assertTrue(comment.startswith('#'))
                self.assertIn('owner:', comment)
                self.assertIn('reason:', comment)

    def test_no_suppression_names_product_or_test_frames(self):
        for _, line in self.entries()[0]:
            with self.subTest(line=line):
                self.assertIsNone(PRODUCT.search(line))

    def test_no_suppression_names_allocator_or_libstdcxx_frames(self):
        for _, line in self.entries()[0]:
            with self.subTest(line=line):
                self.assertIsNone(DENYLIST.search(line))
        for line in ('leak:operator new', 'leak:malloc', 'leak:std::vector', 'leak:__gnu_cxx::pool',
                     'leak:libstdc++.so', 'leak:libc.so.6', 'leak:realloc', 'leak:calloc'):
            with self.subTest(denied=line):
                self.assertIsNotNone(DENYLIST.search(line))

    def test_provenance_records_the_suppressions_hash_not_its_path(self):
        recorded = sanitizer.provenance()
        digest = hashlib.sha256(sanitizer.SUPPRESSIONS.read_bytes()).hexdigest()
        self.assertEqual(set(recorded), {'ASAN_OPTIONS', 'LSAN_OPTIONS', 'UBSAN_OPTIONS'})
        self.assertNotIn('/', recorded['LSAN_OPTIONS'])
        self.assertEqual(recorded['LSAN_OPTIONS'], f'suppressions=sha256:{digest}:print_suppressions=0')
        options = sanitizer.runtime_options()
        for key in ('ASAN_OPTIONS', 'UBSAN_OPTIONS'):
            self.assertEqual(recorded[key], options[key])
        self.assertEqual(sanitizer.recorded(options), recorded)
        self.assertEqual(sanitizer.recorded({}), {})

    def test_thread_recipe_logs_every_report_and_fails_the_process(self):
        log_prefix = Path('/run/0.tsan')
        options = sanitizer.runtime_options('thread', log_prefix)
        self.assertEqual(set(options), {'TSAN_OPTIONS'})
        self.assertIn('TSAN_OPTIONS', sanitizer.PASSTHROUGH)
        self.assertEqual(sanitizer.split_options(options['TSAN_OPTIONS']),
                         ['suppressions=' + sanitizer.option_value(sanitizer.SUPPRESSIONS_TSAN), 'history_size=7',
                          'second_deadlock_stack=1', 'detect_deadlocks=1', 'halt_on_error=0', 'exitcode=66',
                          'log_path=' + sanitizer.option_value(log_prefix)])
        self.assertTrue(sanitizer.SUPPRESSIONS_TSAN.is_absolute())
        with self.assertRaisesRegex(ValueError, 'log prefix'):
            sanitizer.runtime_options('thread')
        recorded = sanitizer.recorded(options)
        self.assertEqual(recorded, sanitizer.provenance('thread'))
        digest = hashlib.sha256(sanitizer.SUPPRESSIONS_TSAN.read_bytes()).hexdigest()
        self.assertEqual(recorded['TSAN_OPTIONS'],
                         f'suppressions=sha256:{digest}:history_size=7:second_deadlock_stack=1:detect_deadlocks=1:'
                         'halt_on_error=0:exitcode=66:log_path=run-directory')

    def test_every_tsan_suppression_names_a_third_party_frame_with_owner_and_reason(self):
        lines = [line.strip() for line in sanitizer.SUPPRESSIONS_TSAN.read_text().splitlines()]
        self.assertTrue(lines[0].startswith('# SPDX-FileCopyrightText:'))
        header = ' '.join(line.lstrip('# ') for line in lines if line.startswith('#'))
        self.assertIn('never names a product frame', header)
        self.assertIn('third-party header templates inlined into our code', header)
        entries = [(index, line) for index, line in enumerate(lines) if line and not line.startswith('#')]
        self.assertTrue(entries)
        for index, line in entries:
            with self.subTest(line=line):
                self.assertRegex(line, r'^(race|called_from_lib|deadlock|mutex):\S+$')
                # ThreadSanitizer aborts when one called_from_lib pattern matches two loaded libraries.
                # A module entry names one library so each prebuilt library carries its own owner and reason.
                if line.startswith('called_from_lib:') or (line.startswith('race:') and '.so' in line):
                    self.assertNotRegex(line, r'[*?\[]')
                self.assertIsNone(PRODUCT.search(line))
                self.assertGreater(index, 0)
                comment = lines[index - 1]
                self.assertTrue(comment.startswith('#'))
                self.assertIn('owner:', comment)
                self.assertIn('reason:', comment)

    def test_usd_race_entries_key_on_anchored_usd_template_frames(self):
        # called_from_lib ignores USD's own allocator calls, so no race entry names a USD module; the USD
        # entries key on USD header templates compiled into our binaries. Anchoring each at the start of the
        # function name keeps it off product functions that only take or return a USD type.
        text = sanitizer.SUPPRESSIONS_TSAN.read_text()
        self.assertTrue(re.findall(r'^called_from_lib:libusd_\w+\.so$', text, re.M))
        self.assertFalse(re.findall(r'^race:.*libusd', text, re.M))
        usd = re.findall(r'^race:(.*pxrInternal.*)$', text, re.M)
        self.assertTrue(usd)
        for pattern in usd:
            with self.subTest(pattern=pattern):
                self.assertRegex(pattern, r'^\^pxrInternal_v0_25_11__pxrReserved__::(Tf|Vt|Sdf|Pcp)\w+(::|<)?$')
                self.assertFalse(template_match(pattern, 'pxrInternal_v0_25_11__pxrReserved__::'
                                                'OmniUsdLiveResolver::_Resolve(std::string const&) const'))
        # No gRPC entry matches the schema probe's file or module, or a product frame that takes a gRPC type.
        grpc = re.findall(r'^race:(.*grpc.*)$', text, re.M)
        self.assertTrue(grpc)
        for pattern in grpc:
            for name in ('/src/tests/native/runtime/grpc_schema_probe.cpp', '/build/live_grpc_schema_probe',
                         'omni::usd::live::Endpoint::open(grpc::ClientContext&, grpc::Status const&)'):
                with self.subTest(pattern=pattern, name=name):
                    self.assertFalse(template_match(pattern, name))
        for pattern, name in (('(*grpc_', 'eventfd_wakeup(grpc_wakeup_fd*)'),
                              ('<grpc::', 'std::shared_ptr<grpc::Channel>::~shared_ptr()'),
                              ('^grpc::', 'grpc::ClientContext::~ClientContext()'), ('^grpc_', 'grpc_shutdown')):
            with self.subTest(pattern=pattern, name=name):
                self.assertIn(pattern, grpc)
                self.assertTrue(template_match(pattern, name))
        # gRPC's internal TLS teardown functions carry C-style names without a gRPC prefix; each entry is anchored
        # at the start and closed by its parameter list, so it names that one function.
        for pattern, name in (('^ssl_protector_destroy(', 'ssl_protector_destroy(tsi_frame_protector*)'),
                              ('^secure_endpoint_unref(',
                               'secure_endpoint_unref((anonymous namespace)::secure_endpoint*)')):
            with self.subTest(pattern=pattern, name=name):
                self.assertIn('race:' + pattern, text.splitlines())
                self.assertTrue(template_match(pattern, name))
                self.assertFalse(template_match(pattern, 'omni::usd::live::' + name))
                self.assertFalse(template_match(pattern, name.replace('(', '_all(', 1)))
        header = ' '.join(line.lstrip('# ') for line in text.splitlines() if line.startswith('#'))
        self.assertIn('never a heap allocation stack or a thread creation stack', header)
        self.assertIn("keyed on a third-party frame in the report's stacks", header)

    def test_path_keyed_and_anchored_entries_name_third_party_frames(self):
        checked = {'path': 0, 'anchored': 0}
        for supp in (sanitizer.SUPPRESSIONS, sanitizer.SUPPRESSIONS_TSAN):
            for line in supp.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                pattern = entry_pattern(line)
                if '/' in pattern and not pattern.startswith('^'):
                    checked['path'] += 1
                if pattern.startswith('^') and pattern.endswith('$'):
                    checked['anchored'] += 1
                with self.subTest(file=supp.name, line=line):
                    self.assertIsNone(suppression_policy_violation(line))
        self.assertTrue(all(checked.values()), checked)  # Both classes exist, so the policy is exercised.
        # Each current entry class passes: OpenSSL source files and directories, an anchored C symbol and
        # the start-anchored USD and gRPC templates the other policy test pins.
        for line in ('leak:/src/crypto/rand/rand_lib.c', 'race:/src/crypto/', 'race:/src/ssl/', 'race:/src/providers/',
                     'leak:^ssl_evp_cipher_fetch$', 'leak:Sdf_ComputeAssetInfoFromIdentifier',
                     'race:^pxrInternal_v0_25_11__pxrReserved__::VtValue::', 'race:^grpc_',
                     'called_from_lib:libusd_sdf.so'):
            with self.subTest(allowed=line):
                self.assertIsNone(suppression_policy_violation(line))
        # Product or unowned paths and anchored C++ names fail.
        for line, reason in (('leak:/source/live/runtime/LiveLayerActor.cpp', 'product source tree'),
                             ('leak:/tests/native/', 'product source tree'),
                             ('race:/tools/live/', 'product source tree'),
                             ('leak:/src/crypto/../../source/live/', 'product source tree'),
                             ('leak:/opt/vendor/other/', 'no third-party source tree'),
                             ('leak:^_ZN4omni3usd4live10LayerActor6submitE$', 'bare C symbol'),
                             ('leak:^omni::usd::live::LayerActor::submit$', 'bare C symbol'),
                             ('leak:^pxrInternal_v0_25_11__pxrReserved__::TfNotice::_Register$', 'bare C symbol'),
                             ('leak:^ssl_*$', 'bare C symbol'),
                             ('leak:^omniUsdLiveRegisterEndpoint$', 'bare C symbol')):
            with self.subTest(denied=line):
                self.assertIn(reason, suppression_policy_violation(line) or '')

    def test_a_suppressions_edit_changes_the_source_fingerprint(self):
        # A replay or CI record whose suppressions differ must not claim the same source.
        original = Path.read_bytes
        baseline = runner.fingerprint()
        for supp in (sanitizer.SUPPRESSIONS, sanitizer.SUPPRESSIONS_TSAN):
            def edited(path, target=supp):
                data = original(path)
                return data + b'# edited\n' if Path(path) == target else data
            with self.subTest(file=supp.name), patch.object(Path, 'read_bytes', edited):
                self.assertNotEqual(runner.fingerprint(), baseline)

    def test_the_source_fingerprint_is_independent_of_the_platform_path_order(self):
        # Windows sorts paths case-insensitively and POSIX does not; every candidate target must agree.
        baseline = runner.fingerprint()
        reversed_order = lambda items, **options: list(reversed(sorted(items, **options)))
        with patch.object(runner, 'sorted', reversed_order, create=True):
            self.assertEqual(runner.fingerprint(), baseline)

    def test_a_passed_suite_with_thread_sanitizer_reports_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            clean = runner.record_sanitizer_reports(dict(outcome='passed'), directory, 3)
            self.assertEqual(clean, dict(outcome='passed', sanitizer_reports=[]))
            (directory / '3.tsan.101').write_text('WARNING: ThreadSanitizer: data race\n')
            (directory / '4.tsan.102').write_text('another suite\n')
            reported = runner.record_sanitizer_reports(dict(outcome='passed'), directory, 3)
            self.assertEqual(reported, dict(outcome='failed', sanitizer_reports=['3.tsan.101'],
                                            reason='ThreadSanitizer reported in 1 process(es)'))
            failed = runner.record_sanitizer_reports(dict(outcome='timeout', reason='watchdog'), directory, 3)
            self.assertEqual((failed['outcome'], failed['reason']), ('timeout', 'watchdog'))

    def test_kind_of_reads_the_recorded_kind(self):
        self.assertEqual(sanitizer.kind_of({'sanitizers': False}), 'none')
        self.assertEqual(sanitizer.kind_of({}), 'none')
        self.assertEqual(sanitizer.kind_of({'sanitizers': True}), 'address')
        self.assertEqual(sanitizer.kind_of({'sanitizers': True, 'sanitizer': 'thread'}), 'thread')

    def test_address_randomization_probe_reads_the_personality(self):
        for value, expected in (('00440000', True), ('00000000', False)):
            with self.subTest(value=value), patch.object(Path, 'read_text', return_value=value + '\n'):
                self.assertIs(sanitizer.address_randomization_disabled(), expected)
        with patch.object(Path, 'read_text', side_effect=OSError('no procfs')):
            self.assertFalse(sanitizer.address_randomization_disabled())

    def test_identity_writers_record_the_sanitizer_kind(self):
        writers = ('write_build_manifest', 'scene_identity', 'plugin_identity', 'wire_identity',
                   'resolver_identity', 'resolver_runtime_identity')
        for name in writers:
            with self.subTest(writer=name):
                usage = subprocess.run([sys.executable, '-B', str(sanitizer.ROOT / 'tools/live' / (name + '.py')),
                                        '--help'], capture_output=True, text=True, timeout=10)
                self.assertEqual(usage.returncode, 0, usage.stderr)
                self.assertIn('--sanitizer {none,address,thread}', usage.stdout)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'input.txt').write_text('synthetic input')
            artifact = root / 'artifact'
            artifact.write_bytes(b'synthetic artifact')
            for arguments, expected in ((['--sanitizers', '0'], (False, 'none')),
                                        (['--sanitizers', '1'], (True, 'address')),
                                        (['--sanitizers', '1', '--sanitizer', 'address'], (True, 'address')),
                                        (['--sanitizers', '1', '--sanitizer', 'thread'], (True, 'thread')),
                                        (['--sanitizers', '0', '--sanitizer', 'none'], (False, 'none')),
                                        (['--sanitizers', '0', '--sanitizer', 'thread'], None),
                                        (['--sanitizers', '1', '--sanitizer', 'none'], None)):
                with self.subTest(arguments=arguments):
                    result = subprocess.run([sys.executable, '-B',
                                             str(sanitizer.ROOT / 'tools/live/write_build_manifest.py'),
                                             '--root', str(root), '--artifact', str(artifact),
                                             '--compiler-id', 'Synthetic', '--compiler-version', '1',
                                             '--configuration', 'Release', '--input', 'input.txt', *arguments],
                                            capture_output=True, text=True, timeout=10)
                    if expected is None:
                        self.assertEqual(result.returncode, 2)
                        continue
                    self.assertEqual(result.returncode, 0, result.stderr)
                    manifest = json.loads(Path(str(artifact) + '.manifest.json').read_text())
                    self.assertEqual((manifest['sanitizers'], manifest['sanitizer']), expected)


if __name__ == '__main__':
    unittest.main()
