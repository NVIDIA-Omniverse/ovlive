# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only packaged diagnostics sample against the independent service ledger."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import unittest

import grpc
import test_scene as scene_support
import tls_fixture
from service import State, Principal, ReferenceServer, pb, rpc


class DiagnosticsSampleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.seed = int(os.environ["LIVE_TEST_SEED"])
        self.artifact = Path(os.environ["LIVE_TEST_ARTIFACT_DIR"])
        self.token = f"private-diagnostics-token-{self.seed}"
        self.base = f"private-diagnostics-base-{self.seed}"
        self.names = [f"private-session-name-{self.seed}-{index}" for index in range(3)]
        self.state = State(self.seed)
        self.state.tokens = {self.token: Principal("sample-fixture-owner")}
        self.processes, self.drainers, self.runs, self.proxies = [], [], [], []
        self.server = ReferenceServer(self.state)
        self.channel = None
        self.addAsyncCleanup(self.cleanup)
        self.address = await self.server.start()
        self.channel = grpc.aio.insecure_channel(self.address)
        await asyncio.wait_for(self.channel.channel_ready(), 5)
        stub = rpc.LiveServiceStub(self.channel)
        for name in self.names:
            await stub.CreateSession(pb.CreateSessionRequest(path=self.base, name=name),
                                     metadata=(("authorization", "Bearer " + self.token),), timeout=5)
        # A different base verifies the sample passes the requested base key.
        await stub.CreateSession(pb.CreateSessionRequest(path=self.base + "-other", name="private-other-name"),
                                 metadata=(("authorization", "Bearer " + self.token),), timeout=5)

    async def cleanup(self):
        failures = []
        for process in self.processes:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        groups = [(process.wait() for process in self.processes), iter(self.drainers)]
        if self.channel is not None:
            groups.append(iter((self.channel.close(),)))
        groups.append((proxy.close() for proxy in self.proxies))
        groups.append(iter((self.server.stop(),)))
        for group in groups:
            try:
                results = await asyncio.wait_for(asyncio.gather(*group, return_exceptions=True), 5)
                failures.extend(repr(result) for result in results if isinstance(result, BaseException))
            except Exception as error:
                failures.append(repr(error))
        ledger = self.state.snapshot()
        ledger.update(seed=self.seed, rpc_attempts=dict(self.state.attempts), sample_runs=self.runs,
                      cleanup_errors=failures)
        path = self.artifact / (self.id().split(".")[-1] + ".diagnostics-sample.json")
        path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        self.state.assert_faults_fired()
        self.assertFalse(failures, failures)

    async def sample(self, extra=(), *, token=None, missing_token=False, arguments=None, default_token=False):
        environment = dict(os.environ)
        environment.pop("LIVE_SERVICES_TOKEN", None)
        environment.pop("LIVE_TEST_SAMPLE_TOKEN", None)
        if not missing_token:
            environment["LIVE_SERVICES_TOKEN" if default_token else "LIVE_TEST_SAMPLE_TOKEN"] = self.token if token is None else token
        args = (["--authority", self.address, "--base-layer-key", self.base,
                 "--token-env", "LIVE_TEST_SAMPLE_TOKEN", "--timeout-ms", "1000"]
                if arguments is None else list(arguments))
        if default_token and arguments is None:
            args[4:6] = []
        process = await asyncio.create_subprocess_exec(os.environ["LIVE_TEST_DIAGNOSTICS_SAMPLE"],
            *args, *extra, env=environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        self.processes.append(process)
        number = len(self.runs)
        record = {"index": number}
        self.runs.append(record)

        async def drain(stream, suffix):
            path = self.artifact / f"{self.id().split('.')[-1]}-{number}.sample.{suffix}"
            remaining, total = 1024 * 1024, 0
            digest = hashlib.sha256()
            with path.open("wb") as output:
                while chunk := await stream.read(16384):
                    total += len(chunk)
                    digest.update(chunk)
                    kept = chunk[:remaining]
                    output.write(kept)
                    remaining -= len(kept)
            record[suffix] = {"bytes": total, "sha256": digest.hexdigest(), "path": path.name}
            self.assertLessEqual(total, 1024 * 1024, "sample exceeded bounded output artifact")
            return path.read_text()

        out = asyncio.create_task(drain(process.stdout, "stdout"))
        err = asyncio.create_task(drain(process.stderr, "stderr"))
        self.drainers.extend((out, err))
        await asyncio.wait_for(process.wait(), 12)
        stdout, stderr = await asyncio.gather(out, err)
        record["exit_code"] = process.returncode
        for secret in (self.token, token, self.base, self.address, *self.names, "private-other-name", "Bearer"):
            if secret:
                self.assertNotIn(secret, stdout + stderr, "sample leaked configuration or resource identity")
        return process.returncode, stdout, stderr

    def summary(self, stderr):
        values = {}
        for name, pattern in (
            ("list", r"^list_sessions result=(\d+) count=(\d+)$"),
            ("endpoint", r"^endpoint_status query=(\d+) result=(\d+) state=(\d+) incident=(\d+)$"),
            ("callbacks", r"^status_callbacks=(\d+) terminal_status_callbacks=(\d+)$"),
            ("sample", r"^sample_result=(\d+) diagnostic_result=(\d+) cleanup_result=(\d+)$"),
        ):
            matches = re.findall(pattern, stderr, re.MULTILINE)
            self.assertEqual(len(matches), 1, f"missing/duplicate {name} summary: {stderr[-4096:]}")
            values[name] = tuple(map(int, matches[0]))
        self.assertGreater(values["callbacks"][0], 0)
        self.assertLessEqual(values["callbacks"][1], values["callbacks"][0])
        self.assertEqual(values["endpoint"][0], 0)
        self.assertEqual(values["sample"][1:], (0, 0))
        return values

    def json_output(self, stdout, *, snapshot=True):
        decoder, documents, offset = json.JSONDecoder(), [], 0
        while offset < len(stdout):
            if stdout[offset].isspace():
                offset += 1
                continue
            document, offset = decoder.raw_decode(stdout, offset)
            documents.append(document)
        self.assertTrue(documents)
        exported = documents.pop() if snapshot else None
        if snapshot:
            self.assertIn("metrics", exported, "snapshot must follow listener retirement")
        self.assertTrue(documents, "expected native diagnostic callback records")
        expected = set("text_encoding struct_size abi_version schema_version event_code event_name severity "
                       "timestamp_ms elapsed_ms runtime_instance_id sequence endpoint subject_id incident_id "
                       "field_count fields".split())
        for item in documents:
            self.assertEqual(set(item), expected)
            self.assertEqual(item["text_encoding"], "escaped_bytes")
            self.assertEqual(item["schema_version"], 1)
            self.assertGreater(item["struct_size"], 0)
            self.assertGreater(item["abi_version"], 0)
            self.assertGreater(item["runtime_instance_id"], 0)
            self.assertGreater(item["event_code"], 0)
            self.assertRegex(item["event_name"], r"^[a-z_]+$")
            self.assertIn(item["severity"], range(5))
            self.assertEqual(item["field_count"], len(item["fields"]))
            for field in item["fields"].values():
                self.assertEqual(set(field), {"kind", "value"})
                self.assertIn(field["kind"], (1, 2, 3, 4))
                self.assertIs(type(field["value"]), {1: int, 2: int, 3: str, 4: bool}[field["kind"]])
                if field["kind"] == 1:
                    self.assertGreaterEqual(field["value"], 0)
        sequences = [item["sequence"] for item in documents]
        self.assertEqual(sequences, sorted(set(sequences)))
        self.assertEqual(len({item["runtime_instance_id"] for item in documents}), 1)
        return documents, exported

    def check_snapshot(self, snapshot, records, result):
        self.assertEqual(snapshot["schema_version"], 1)
        self.assertEqual(snapshot["runtime_instance_id"], records[0]["runtime_instance_id"])
        self.assertGreater(snapshot["provenance"]["abi_version"], 0)
        self.assertRegex(snapshot["provenance"]["proto_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(snapshot["provenance"]["source_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(snapshot["effective_configuration"])
        self.assertTrue(snapshot["statuses"])
        metrics = snapshot["metrics"]
        self.assertEqual((metrics["rpc_attempts"], metrics["rpc_completions"], metrics["rpc_in_flight"]), (1, 1, 0))
        self.assertEqual(metrics["rpc_by_operation_result"], [{"operation": 10, "result": result, "count": 1}])
        self.assertEqual(metrics["rpc_attempts_by_operation"], [{"operation": 10, "count": 1}])
        self.assertEqual(metrics["rpc_duration_ms"]["bounds"], [1, 5, 10, 50, 100, 500, 1000, 5000])
        self.assertEqual(len(metrics["rpc_duration_ms"]["counts"]), 9)
        self.assertEqual(sum(metrics["rpc_duration_ms"]["counts"]), 1)
        duration = metrics['rpc_duration_ms_by_operation_result']
        self.assertEqual(duration['bounds'], metrics['rpc_duration_ms']['bounds'])
        self.assertEqual(len(duration['entries']), 1)
        row = duration['entries'][0]
        self.assertEqual((row['operation'], row['result'], row['count']), (10, result, 1))
        self.assertEqual(row['counts'], metrics['rpc_duration_ms']['counts'])
        self.assertIs(type(row['sum_ms']), int)
        self.assertGreaterEqual(row['sum_ms'], 0)
        for name in ("invalid", "history_overwritten", "history_dropped", "sink_failures",
                     "sinks_disabled", "closed_incidents_evicted"):
            self.assertEqual(metrics[name], 0, name)
        # Unregistration may cancel still-queued log records. Those losses are
        # explicit; operation outcome remains available in history and metrics.
        self.assertIs(type(metrics["sink_dropped"]), int)
        self.assertGreaterEqual(metrics["sink_dropped"], 0)
        history = snapshot["recent_records"]
        self.assertTrue(history)
        completions = [record for record in history if record['event_name'] in ('rpc_completed', 'rpc_failed')]
        self.assertEqual(len(completions), 1)
        elapsed = completions[0]['fields']['duration_ms']
        self.assertEqual(row['sum_ms'], elapsed)
        expected_buckets = [0] * 9
        expected_buckets[sum(elapsed > bound for bound in duration['bounds'])] = 1
        self.assertEqual(row['counts'], expected_buckets)
        sequences = [item["sequence"] for item in history]
        self.assertEqual(sequences, sorted(set(sequences)))
        self.assertLessEqual(snapshot["history"]["first_sequence"], sequences[0])
        self.assertGreaterEqual(snapshot["history"]["last_sequence"], sequences[-1])
        self.assertIs(snapshot["history"]["truncated"], False)

    async def test_json_records_and_snapshot_count_read_only_list(self):
        before = self.state.snapshot()["commits"]
        code, stdout, stderr = await self.sample(("--debug", "--snapshot"))
        self.assertEqual(code, 0, stderr)
        summary = self.summary(stderr)
        self.assertEqual(summary["list"], (0, 3))
        self.assertEqual(summary["endpoint"][:3], (0, 0, 2))
        records, snapshot = self.json_output(stdout)
        self.check_snapshot(snapshot, records, 0)
        completed = [record for record in snapshot["recent_records"] if record["event_name"] == "rpc_completed"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["fields"]["operation"], 10)
        self.assertEqual(completed[0]["fields"]["result"], 0)
        self.assertEqual(self.state.attempts, {"CreateSession": 4, "ListSessions": 1})
        self.assertEqual(self.state.snapshot()["commits"], before)

    async def test_tls_proxy_with_private_ca(self):
        pki = await tls_fixture.case_pki(self)
        proxy = scene_support.RotatingProxy(self.address, ssl_context=pki.server_context())
        self.proxies.append(proxy)
        address = await proxy.start()
        code, stdout, stderr = await self.sample(arguments=[
            '--authority', address, '--base-layer-key', self.base,
            '--token-env', 'LIVE_TEST_SAMPLE_TOKEN', '--timeout-ms', '1000',
            '--ca-bundle', str(pki.ca), '--tls-server-name', tls_fixture.NAME,
            '--transport-security', 'tls', '--debug'])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.summary(stderr)['list'], (0, 3))

    async def test_text_format_is_single_line_typed_and_read_only(self):
        before = self.state.snapshot()["commits"]
        code, stdout, stderr = await self.sample(("--format", "text", "--debug"), default_token=True)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.summary(stderr)["list"], (0, 3))
        lines = stdout.splitlines()
        self.assertTrue(lines)
        sequences, runtimes = [], []
        for line in lines:
            self.assertRegex(line, r'^(TRACE|DEBUG|INFO|WARN|ERROR) "[a-z_]+" text_encoding=escaped_bytes ')
            self.assertRegex(line, r' field_count=\d+ fields=\{.*\}$')
            sequences.append(int(re.search(r" sequence=(\d+) ", line).group(1)))
            runtimes.append(int(re.search(r" runtime_instance_id=(\d+) ", line).group(1)))
        self.assertEqual(sequences, sorted(set(sequences)))
        self.assertEqual(len(set(runtimes)), 1)
        self.assertTrue(any(re.search(r'"[a-z_]+"\[1\]=\d+', line) for line in lines))
        self.assertEqual(self.state.attempts, {"CreateSession": 4, "ListSessions": 1})
        self.assertEqual(self.state.snapshot()["commits"], before)

    async def test_permission_failure_retains_structured_error_history(self):
        self.state.plan("ListSessions", "before", code=grpc.StatusCode.PERMISSION_DENIED)
        before = self.state.snapshot()["commits"]
        code, stdout, stderr = await self.sample(("--debug", "--snapshot"))
        self.assertEqual(code, 1)
        self.assertEqual(self.summary(stderr)["list"], (7, 0))
        records, snapshot = self.json_output(stdout)
        self.check_snapshot(snapshot, records, 7)
        failed = [record for record in snapshot["recent_records"] if record["event_name"] == "rpc_failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["fields"]["result"], 7)
        self.assertEqual(failed[0]["fields"]["grpc_status"], 7)
        self.assertTrue(any(item["event_name"] == "rpc_failed" for item in snapshot["recent_records"]))
        self.assertEqual(self.state.attempts, {"CreateSession": 4, "ListSessions": 1})
        self.assertEqual(self.state.snapshot()["commits"], before)

    async def test_blocked_merged_output_does_not_block_rpc_and_watchdog_exits(self):
        before = self.state.snapshot()["commits"]
        read_fd, write_fd = os.pipe()
        try:
            os.set_blocking(write_fd, False)
            filled = 0
            while True:
                try:
                    filled += os.write(write_fd, b"P" * 4096)
                    self.assertLessEqual(filled, 1024 * 1024)
                except BlockingIOError:
                    break
            self.assertGreater(filled, 0)
            os.set_blocking(write_fd, True)
            environment = dict(os.environ, LIVE_SERVICES_TOKEN=self.token)
            process = await asyncio.create_subprocess_exec(os.environ["LIVE_TEST_DIAGNOSTICS_SAMPLE"],
                "--authority", self.address, "--base-layer-key", self.base, "--timeout-ms", "200",
                "--debug", "--snapshot", env=environment, stdout=write_fd, stderr=write_fd)
            self.processes.append(process)
            os.close(write_fd)
            write_fd = None
            # Deliberately do not read: neither diagnostic stdout nor main's
            # merged stderr/snapshot output can make progress. Only the sample's
            # own absolute watchdog may terminate this successful test.
            await asyncio.wait_for(process.wait(), 4)
            self.assertEqual(process.returncode, 3)
            output = bytearray()
            while chunk := os.read(read_fd, 16384):
                output.extend(chunk)
                self.assertLessEqual(len(output), 1024 * 1024)
            path = self.artifact / (self.id().split(".")[-1] + ".blocked-output.log")
            path.write_bytes(output)
            self.runs.append({"exit_code": process.returncode, "prefilled_bytes": filled,
                              "merged_output": {"bytes": len(output), "sha256": hashlib.sha256(output).hexdigest(),
                                                "path": path.name}})
            self.assertEqual(output, b"P" * filled)
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)
        self.assertEqual(self.state.attempts, {"CreateSession": 4, "ListSessions": 1})
        self.assertEqual(self.state.snapshot()["commits"], before)

    async def test_invalid_cli_and_missing_token_fail_before_service_calls(self):
        before = dict(self.state.attempts)
        standard = ["--authority", self.address, "--base-layer-key", self.base,
                    "--token-env", "LIVE_TEST_SAMPLE_TOKEN"]
        cases = [([], False), (standard, True),
                 (standard + ["--timeout-ms", "0"], False),
                 (standard + ["--timeout-ms", "60001"], False),
                 (standard + ["--transport-security", "invalid"], False),
                 (standard + ["--format", "private-invalid-format"], False),
                 (standard + ["--unknown-private-option", "private-argument"], False)]
        for arguments, missing in cases:
            with self.subTest(arguments=arguments[-2:], missing_token=missing):
                code, stdout, stderr = await self.sample(arguments=arguments, missing_token=missing)
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertTrue(stderr)
                for private in ("private-invalid-format", "--unknown-private-option", "private-argument"):
                    self.assertNotIn(private, stderr)
                self.assertEqual(self.state.attempts, before)
