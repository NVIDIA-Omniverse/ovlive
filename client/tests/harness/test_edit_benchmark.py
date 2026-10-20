# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.live import edit_benchmark as bench
from tools.live import edit_benchmark_clients as clients
from tools.live.wire_identity import tree_hash


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_tree(root):
    """A synthetic extension inside its plugin package, with the identity manifest the build writes."""
    extension = root / 'plugin/omni_usd_live/python/omni/usd/live/_live.so'
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b'extension')
    plugin = root / 'plugin/omni_usd_live/lib/libomni_usd_live.so'
    plugin.parent.mkdir(parents=True)
    plugin.write_bytes(b'plugin')
    driver = root / 'plugin-driver'
    driver.write_bytes(b'driver')
    usd_lib = root / 'usd/lib'
    usd_lib.mkdir(parents=True)
    (usd_lib / 'libusd_sdf.so').write_bytes(b'sdf')
    manifest = dict(schema_version=1, boundary='python_adapter', artifact_sha256=sha256(extension),
                    plugin_driver=str(driver),
                    plugin_identity=dict(artifact_sha256=sha256(driver),
                                         dependencies=[dict(role='plugin', path=str(plugin), sha256=sha256(plugin))],
                                         dependency_trees=[dict(role='usd_lib', path=str(usd_lib),
                                                                sha256=tree_hash(usd_lib))]))
    Path(str(extension) + '.manifest.json').write_text(json.dumps(manifest))
    return extension, manifest


class EditBenchmarkMetricTests(unittest.TestCase):
    def test_a_turn_is_visible_at_the_first_update_holding_it_or_a_later_turn(self):
        edits = [(0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0)]
        # The second update coalesces turns 1 and 2; the snapshot marker None is not a turn.
        received = [(0.5, 1, None, 10), (1.5, 2, 0.0, 10), (4.0, 3, 2.0, 10), (5.0, 4, 3.0, 10)]
        self.assertEqual(bench.visibility(edits, received), {0: 1.5, 1: 4.0, 2: 4.0, 3: 5.0})
        metrics = bench.peer_metrics(edits, received)
        self.assertEqual(metrics['latency_ms'], dict(p50=2000.0, p95=3000.0, p99=3000.0, max=3000.0))
        self.assertEqual((metrics['visible_turns'], metrics['updates'], metrics['turns_per_update']), (4, 3, 4 / 3))
        self.assertEqual(metrics['max_backlog_turns'], 3)  # At turn 3's edit only turn 0 is visible.
        self.assertEqual(metrics['throughput_turns_per_s'], 4 / 5.0)
        self.assertEqual(metrics['drain_ms'], 2000.0)

    def test_a_turn_never_seen_leaves_throughput_and_drain_unset(self):
        metrics = bench.peer_metrics([(0, 0.0), (1, 1.0)], [(0.5, 1, 0.0, 10)])
        self.assertEqual((metrics['visible_turns'], metrics['throughput_turns_per_s'], metrics['drain_ms']),
                         (1, None, None))
        self.assertEqual(metrics['max_backlog_turns'], 1)

    def test_latency_splits_at_the_commit_of_the_update_that_made_a_turn_visible(self):
        edits = [(0, 0.0), (1, 1.0)]
        received = [(3.0, 2, 1.0, 10)]  # One update holds both turns.
        split = bench.split_at_service(edits, received, [(2, 2.5)])
        self.assertEqual(split['edit_to_commit_ms'], dict(p50=1500.0, p99=2500.0))
        self.assertEqual(split['commit_to_peer_ms'], dict(p50=500.0, p99=500.0))

    def test_nearest_rank_percentile(self):
        self.assertEqual(bench.percentile(list(range(1, 101)), .99), 99)
        self.assertEqual(bench.percentile([5], .5), 5)
        self.assertIsNone(bench.percentile([], .5))

    @staticmethod
    def case(scenario, client, latency, throughput, backlog, p95=None, rejected=0, visible=10, drained=True,
             failures=(), version='2.74.0'):
        peer = dict(latency_ms=dict(p50=latency, p95=p95 or latency, p99=latency), turns=10 - rejected,
                    visible_turns=visible - rejected, throughput_turns_per_s=throughput, max_backlog_turns=backlog,
                    authored_turns_per_s=60.0, updates=10, turns_per_update=1.0, drain_ms=None)
        writer = dict(accepted_turns=10 - rejected, rejected_turns=rejected, first_rejected_turn=10 - rejected,
                      drained=drained)
        if client == 'ovlive':
            writer.update(peak_pending_write_count=1, peak_buffered_edit_count=1, blocked_turns=0)
        else:
            writer.update(failures=list(failures), version=version, peak_in_flight=1, window=16, peak_buffered_turns=1)
        return dict(scenario=scenario, rtt_ms=50, client=client, metrics=dict(peer=peer, writer=writer))

    def test_the_bar_is_latency_everywhere_throughput_when_sustained_and_every_turn_delivered(self):
        case = self.case
        cases = [case('single', 'ovlive', 50.0, 60.0, 9, p95=99.0), case('single', 'client-library', 50.0, 60.0, 4),
                 case('multi', 'ovlive', 40.0, 59.0, 3), case('multi', 'client-library', 50.0, 60.0, 4),
                 case('burst', 'ovlive', 40.0, 10.0, 9), case('burst', 'client-library', 50.0, 60.0, 4),
                 case('multi-burst', 'ovlive', 60.0, 90.0, 3), case('multi-burst', 'client-library', 50.0, 60.0, 4),
                 case('batch-in-one-frame', 'ovlive', 40.0, 90.0, 3, rejected=2),
                 case('batch-in-one-frame', 'client-library', 50.0, 60.0, 4)]
        verdicts = {v['scenario']: v['meets'] for v in bench.comparisons(cases)}
        # p95 and backlog never decide; burst throughput never decides; a refused change block does.
        self.assertEqual(verdicts, {'single': True, 'multi': False, 'burst': True, 'multi-burst': False,
                                    'batch-in-one-frame': False})

    def test_an_incomplete_delivery_on_either_side_misses_the_bar(self):
        case = self.case
        baseline = case('burst', 'client-library', 50.0, 60.0, 4)
        incomplete = {
            'lost turns': ([case('burst', 'ovlive', 40.0, None, 1, visible=9), baseline], ['ovlive']),
            'not drained': ([case('burst', 'ovlive', 40.0, 90.0, 1, drained=False), baseline], ['ovlive']),
            'refused turns': ([case('multi-burst', 'ovlive', 40.0, 90.0, 1, rejected=3),
                               case('multi-burst', 'client-library', 50.0, 60.0, 4)], ['ovlive']),
            'failed baseline write': ([case('burst', 'ovlive', 40.0, 90.0, 1),
                                       case('burst', 'client-library', 50.0, 60.0, 4, failures=['Result.ERROR'])],
                                      ['baseline']),
            'baseline lost turns': ([case('single', 'ovlive', 40.0, 90.0, 1),
                                     case('single', 'client-library', 50.0, None, 4, visible=8)], ['baseline']),
        }
        for name, (cases, sides) in incomplete.items():
            with self.subTest(name):
                [verdict] = bench.comparisons(cases)
                self.assertFalse(verdict['meets'])
                self.assertEqual(verdict['incomplete'], sides)
                report = dict(seed=42, service='opaque', jitter_ms=2.0, labels={}, cases=cases,
                              identities=bench.identities(None, None, None, None))
                self.assertIn(f"| no ({sides[0]} incomplete) |", bench.summary(report))
        [complete] = bench.comparisons([case('burst', 'ovlive', 40.0, 90.0, 1), baseline])
        self.assertEqual((complete['meets'], complete['incomplete']), (True, []))
        self.assertEqual(complete['delivery']['ovlive'], dict(authored_turns=10, visible_turns=10, refused_turns=0,
                                                              failed_writes=0, drained=True, drain_timed_out=False,
                                                              complete=True))

    def test_only_the_documented_baseline_release_can_meet_the_bar(self):
        case = self.case
        for version, qualified in (('2.74.0', True), ('2.74.0-release.1+gl.0', True), ('2.74.0+gl.0', True),
                                   ('2.75.0', False), ('2.74.01', False), (None, False)):
            with self.subTest(version):
                cases = [case('single', 'ovlive', 40.0, 60.0, 1), case('single', 'client-library', 50.0, 60.0, 4,
                                                                         version=version)]
                [verdict] = bench.comparisons(cases)
                self.assertEqual((verdict['qualified'], verdict['meets'], verdict['baseline_version']),
                                 (qualified, qualified, version))
                report = dict(seed=42, service='opaque', jitter_ms=2.0, labels={}, cases=cases,
                              identities=bench.identities(None, None, None, None))
                self.assertIn('| yes |' if qualified else f'| no (baseline {version}, not 2.74.0) |',
                              bench.summary(report))
        cases = [case('single', 'ovlive', 40.0, 60.0, 1, visible=9),
                 case('single', 'client-library', 50.0, 60.0, 4, version='2.75.0')]
        report = dict(seed=42, service='opaque', jitter_ms=2.0, labels={}, cases=cases,
                      identities=bench.identities(None, None, None, None))
        self.assertIn('| no (ovlive incomplete; baseline 2.75.0, not 2.74.0) |', bench.summary(report))

    @staticmethod
    def ovlive_writer(state, drained):
        return dict(client='ovlive', edits=[(0, 0.0), (1, 1.0)], rejected=[], frame_timeouts=0, drained=drained,
                    blocked_turns=0, peak=dict(pending_write_count=1, buffered_edit_count=1),
                    final=dict(state=state, result=13 if state == 7 else 0, pending_write_count=0,
                               buffered_edit_count=0), delivery_status=0, metrics={})

    def test_a_failed_ovlive_layer_is_a_failed_write_and_not_a_drain_timeout(self):
        received = [(0.5, 1, 0.0, 10)]
        failed = bench.analyze(dict(writer=self.ovlive_writer(7, False), neutral=dict(received=received), service={}))
        self.assertEqual(failed['writer']['failures'], ['layer FAILED, result 13'])
        self.assertEqual(failed['delivery'], dict(authored_turns=2, visible_turns=1, refused_turns=0, failed_writes=1,
                                                  drained=False, drain_timed_out=False, complete=False))
        timed_out = bench.analyze(dict(writer=self.ovlive_writer(2, False), neutral=dict(received=received),
                                       service={}))
        self.assertEqual((timed_out['delivery']['failed_writes'], timed_out['delivery']['drain_timed_out']), (0, True))
        report = dict(seed=42, service='opaque', jitter_ms=2.0, labels={}, identities=bench.identities(None, None, None, None),
                      cases=[dict(scenario='single', rtt_ms=0, client='ovlive', metrics=failed),
                             dict(scenario='single', rtt_ms=20, client='ovlive', metrics=timed_out)])
        text = bench.summary(report)
        self.assertIn('peer saw 1/2 turns, 1 failed writes, not drained', text)
        self.assertIn('peer saw 1/2 turns, 0 failed writes, drain timed out', text)

    def test_a_case_records_its_delivery(self):
        raw = dict(writer=dict(client='client-library', edits=[(0, 0.0), (1, 1.0)], updates=[], failures=[],
                               drained=True, window=16, peak=dict(in_flight=1, buffered_turns=1), version='2.74.0'),
                   neutral=dict(received=[(0.5, 1, 0.0, 10)]), service={})
        self.assertEqual(bench.analyze(raw)['delivery'], dict(authored_turns=2, visible_turns=1, refused_turns=0,
                                                              failed_writes=0, drained=True, drain_timed_out=False,
                                                              complete=False))

    def test_service_counts_come_from_the_full_ledger(self):
        def entry(operation, writer='w', layer=bench.LAYER, session=bench.SESSION, size=3):
            return SimpleNamespace(operation=operation, session=session, layer=layer, writer=writer, payload=b'x' * size)
        # More updates than the graph service's layer retains; other layers, sessions and operations do not count.
        ledger = [entry('CreateLayer', writer='', size=100)] + [entry('UpdateLayer') for _ in range(100)] + [
            entry('UpdateLayer', writer='v', size=5), entry('UpdateLayer', layer='other.live'),
            entry('UpdateLayer', session='other')]
        self.assertGreater(100, bench.GRAPH_RETAINED_UPDATES)
        self.assertEqual(bench.update_counts(ledger), dict(commits=101, writers=[dict(commits=100, bytes=300),
                                                                                  dict(commits=1, bytes=5)]))

    def test_the_report_records_the_measured_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extension, manifest = build_tree(root)
            package = root / 'client-library'
            module = package / 'release/bindings-python/omni/client'
            module.mkdir(parents=True)
            (module / '__init__.py').write_text("def get_version():\n    return '2.74.0-release.1+gl.0'\n")
            measured = bench.identities(extension, bench.extension_manifest(extension), package,
                                        bench.client_library_version(package))
        self.assertEqual(measured, dict(ovlive=dict(python_extension=str(extension), manifest=manifest),
                                        client_library=dict(package=str(package), version='2.74.0-release.1+gl.0')))
        self.assertIn('Omniverse Client Library 2.74.0-release.1+gl.0', bench.identity_line(measured))
        report = dict(identities=measured, cases=[self.case('single', 'client-library', 50.0, 60.0, 4)])
        with self.assertRaisesRegex(RuntimeError, 'differs from the recorded'):
            bench.check_client_library_versions(report)
        report['cases'][0]['metrics']['writer']['version'] = '2.74.0-release.1+gl.0'
        bench.check_client_library_versions(report)

    def test_the_extension_manifest_must_describe_the_loaded_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extension, manifest = build_tree(root)
            self.assertEqual(bench.extension_manifest(extension), manifest)
            plugin = Path(manifest['plugin_identity']['dependencies'][0]['path'])
            tree = Path(manifest['plugin_identity']['dependency_trees'][0]['path'])
            changes = {'extension differs': lambda: extension.write_bytes(b'rebuilt'),
                       'plugin differs': lambda: Path(manifest['plugin_driver']).write_bytes(b'rebuilt'),
                       'plugin plugin differs': lambda: plugin.write_bytes(b'rebuilt'),
                       'SDK usd_lib differs': lambda: (tree / 'libusd_usd.so').write_bytes(b'added')}
            for message, change in changes.items():
                with self.subTest(message):
                    shutil.rmtree(root)
                    root.mkdir()
                    extension, manifest = build_tree(root)
                    change()
                    with self.assertRaisesRegex(ValueError, message):
                        bench.extension_manifest(extension)
            Path(str(extension) + '.manifest.json').write_text(json.dumps(dict(manifest, boundary='plugin')))
            with self.assertRaisesRegex(ValueError, 'not a Python binding identity manifest'):
                bench.extension_manifest(extension)

    def test_both_usd_hosts_clear_the_invoking_shells_usd_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            extension, manifest = build_tree(Path(directory))
            shell = dict(PXR_PLUGINPATH_NAME='/shell/plugins', PXR_WORK_THREAD_LIMIT='1', AR_DEFAULT_RESOLVER='Shell',
                         PXR_AR_DEFAULT_SEARCH_PATH='/shell', OMNI_X='1', PYTHONPATH='/shell/python')
            with patch.dict(os.environ, shell):
                ovlive = bench.extension_environment(extension)
                baseline = bench.baseline_usd_environment(Path(directory) / 'client-library', extension)
        for environment in (ovlive, baseline):
            self.assertEqual({k: v for k, v in environment.items() if k.startswith(('PXR_', 'AR_'))},
                             dict(PXR_PLUGINPATH_NAME='', PXR_AR_DEFAULT_SEARCH_PATH='', AR_DEFAULT_RESOLVER=''))
        self.assertEqual(baseline['LD_LIBRARY_PATH'], ovlive['LD_LIBRARY_PATH'])
        self.assertNotIn('OMNI_X', baseline)
        self.assertTrue(baseline['PYTHONPATH'].startswith(str(Path(directory) / 'client-library/release/bindings-python')))

    def test_the_baseline_host_scene_matches_the_seeded_live_layer(self):
        for prims, attributes in ((1, 1), (50, 3), (10000, 1)):
            with self.subTest(prims=prims):
                _nodes, paths, _sections = bench.layout(prims, attributes)
                self.assertEqual([f'/{prim}.{name}' for prim, names in clients.scene(prims, attributes)
                                  for name in names], paths)

    def test_capability_fields_are_length_delimited_protobuf(self):
        self.assertEqual(bench.field(1, b'ab'), b'\x0a\x02ab')
        self.assertEqual(bench.field(2, b'x' * 200)[:3], b'\x12\xc8\x01')


class ChildTests(unittest.IsolatedAsyncioTestCase):
    async def test_library_lines_and_child_stderr_both_reach_the_log(self):
        script = ('import sys\n'
                  'print("library line", flush=True)\n'
                  'print(\'{"type": "ready"}\', flush=True)\n'
                  'sys.stdin.readline()\n'
                  'sys.stderr.write("child stderr\\n")\n')
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'child.log'
            child = await bench.Child.start('child', [sys.executable, '-c', script], log)
            try:
                await child.expect('ready', 60)  # Copies the library line to the log first.
                child.send('go')
                await asyncio.wait_for(child.process.wait(), 60)
            finally:
                await child.close()
            self.assertEqual(log.read_text().splitlines(), ['library line', 'child stderr'])

    async def test_a_child_that_keeps_logging_still_times_out(self):
        class Stdout:
            reads = 0

            async def readline(self):
                Stdout.reads += 1
                return b'library line\n'
        clock = iter([0.0, 0.0, 1.0, 2.0, 10.0])
        with open(os.devnull, 'ab') as handle:
            child = bench.Child('child', SimpleNamespace(stdout=Stdout()), Path(os.devnull), handle)
            with patch.object(bench, 'time', SimpleNamespace(monotonic=lambda: next(clock))):
                with self.assertRaisesRegex(RuntimeError, 'sent no JSON record within 5 s'):
                    await child.receive(5)
        self.assertEqual(Stdout.reads, 3)


if __name__ == '__main__':
    unittest.main()
