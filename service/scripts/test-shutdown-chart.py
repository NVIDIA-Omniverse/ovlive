#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline graceful-shutdown chart regressions. Requires Helm and PyYAML; never
contacts a cluster.

The chart's shutdown settings must agree with the binary's:

* the default render passes, with the paced stream-ending settings, the
  termination grace, the rollout strategy and the derived minReadySeconds;
* a termination grace below the minimum a graceful shutdown can take fails
  the render, naming the values it came from;
* that minimum covers the binary's own (`grpc::transport::minimum_termination_grace`)
  and exceeds it, from rounding each term up to whole seconds, by at most a
  second: the chart's two 6 s terms are the GOAWAY head start plus the close
  grace, and the final flush bound, read from `src/grpc/transport.rs`;
* the check computes with the digits the binary receives, so a quoted
  setting or grace with surrounding space counts at its number, and a
  minReadySeconds that is not a whole number fails the render naming it;
* a values-file grace of a million or more renders on the pod as whole
  digits, not `2e+06`;
* the paced-ending defaults equal the binary's (`src/config.rs`);
* the default rollout takes at most 3 steps at any replica count from 1 to
  99, moving one pod at a time at 3 replicas or fewer, with the surge
  headroom the docs state.
"""

import math
from pathlib import Path
import re
import subprocess
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SERVICE_CHART = ROOT / "helm/live-edit-service"
TRANSPORT_RS = ROOT / "src/grpc/transport.rs"
CONFIG_RS = ROOT / "src/config.rs"
RELEASE = "live-edit-service"


def render(values=None, *, error=None):
    command = ["helm", "template", RELEASE, str(SERVICE_CHART),
               "--namespace", "live-edit", "-f", "-"]
    result = subprocess.run(command, input=yaml.safe_dump(values or {}),
                            capture_output=True, text=True, check=False)
    if error is not None:
        if result.returncode == 0:
            raise AssertionError(f"Expected the render to fail naming {error!r}; it rendered")
        return result.stderr
    if result.returncode:
        raise AssertionError(result.stderr)
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def server_deployment(documents):
    """The server's Deployment, by name and its pods' component=server label:
    the default render also has the scaler's Deployment."""
    matches = [doc for doc in documents if doc["kind"] == "Deployment"
               and doc["metadata"]["name"] == RELEASE
               and doc["spec"]["template"]["metadata"]["labels"].get(
                   "app.kubernetes.io/component") == "server"]
    assert len(matches) == 1, len(matches)
    return matches[0]


def config_data(documents):
    """The data of the ConfigMap that carries the service's env vars."""
    maps = [doc for doc in documents if doc["kind"] == "ConfigMap"
            and "LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS" in (doc.get("data") or {})]
    assert len(maps) == 1, len(maps)
    return maps[0]["data"]


def duration_secs(source, name):
    """A `pub const NAME: Duration = Duration::from_{secs,millis}(N);` in seconds."""
    match = re.search(
        rf"pub const {name}: Duration = Duration::from_(secs|millis)\((\d+)\);", source)
    assert match, f"{name} not found in {TRANSPORT_RS}"
    value = int(match.group(2))
    return value if match.group(1) == "secs" else value / 1000


def binary_default(source, var):
    match = re.search(rf'optional_parsed\(\s*"{var}",\s*(\d+)u32,?\s*\)', source)
    assert match, f"{var} default not found in {CONFIG_RS}"
    return int(match.group(1))


TRANSPORT = TRANSPORT_RS.read_text()
GOAWAY_HEAD_START = duration_secs(TRANSPORT, "STREAM_END_DELAY_AFTER_GOAWAY")
CLOSE_GRACE = duration_secs(TRANSPORT, "STREAM_CLOSE_GRACE")
FLUSH_BOUND = duration_secs(TRANSPORT, "POST_BUDGET_FLUSH_BOUND")


def binary_minimum_grace(timeout_secs, deadline_ms, rate, max_window_secs, overall_secs):
    """`minimum_termination_grace` as the binary computes it, in whole seconds:
    drain + GOAWAY head start + max(deadline, window) + close grace + overall
    budget + flush bound, the window at its cap (0 when either pacing switch
    is 0)."""
    window = 0 if rate == 0 or max_window_secs == 0 else max_window_secs
    total = (timeout_secs + GOAWAY_HEAD_START + max(deadline_ms / 1000, window)
             + CLOSE_GRACE + overall_secs + FLUSH_BOUND)
    return math.ceil(total)


def chart_minimum_grace(timeout_secs, deadline_ms, rate, max_window_secs, overall_secs):
    """The minimum the chart enforces: shutdown.timeoutSecs +
    max(ceil(rpc.deadlineMs / 1000), the window) + ceil(GOAWAY head start +
    close grace) + shutdown.overallBudgetSecs + the flush bound, in whole
    seconds. Each term rounded up on its own, so it is never below the binary's
    minimum and at most a second above it."""
    window = 0 if rate == 0 else max_window_secs
    return (timeout_secs + max(math.ceil(deadline_ms / 1000), window)
            + math.ceil(GOAWAY_HEAD_START + CLOSE_GRACE) + overall_secs + FLUSH_BOUND)


def shutdown_values(grace=None, *, timeout=5, deadline_ms=30000, rate=25, window=20,
                    overall=30):
    values = {"shutdown": {"timeoutSecs": timeout, "overallBudgetSecs": overall,
                           "streamEndRate": rate, "streamEndMaxWindowSecs": window},
              "rpc": {"deadlineMs": deadline_ms}}
    if grace is not None:
        values["terminationGracePeriodSeconds"] = grace
    return values


def surge_pods(max_surge, replicas):
    """maxSurge resolved as the Deployment controller does
    (`ResolveFenceposts`): an integer as is, a percentage of the replicas
    rounded up."""
    if isinstance(max_surge, int):
        return max_surge
    percent = int(max_surge.rstrip("%"))
    return -(-percent * replicas // 100)


class ShutdownChartTests(unittest.TestCase):
    def test_the_chart_constants_equal_the_binarys(self):
        # The chart's first 6 s term: the 0.25 s GOAWAY head start and the 5 s
        # close grace after the bound after GOAWAY, rounded up together.
        self.assertEqual(math.ceil(GOAWAY_HEAD_START + CLOSE_GRACE), 6)
        # The chart's second 6 s term: the final flush's cap.
        self.assertEqual(FLUSH_BOUND, 6)
        # The formula's six terms, each setting read from the rendered
        # ConfigMap ($env), as the binary receives it.
        deployment = (SERVICE_CHART / "templates/deployment.yaml").read_text()
        for pattern in (
                r'\$timeoutSecs := index \$env "LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS"',
                r'\$overallBudgetSecs := index \$env "LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS"',
                r'\$deadlineSecs := div \(add \(int64 \(index \$env "LIVE_EDIT_RPC_DEADLINE_MS"\)\) '
                r'999\) 1000',
                r'\$streamEndWindowSecs := int64 \(index \$env '
                r'"LIVE_EDIT_SHUTDOWN_STREAM_END_MAX_WINDOW_SECS"\)',
                r"\$minGrace := add \(int64 \$timeoutSecs\) "
                r"\(max \$deadlineSecs \$streamEndWindowSecs\) 6 "
                r"\(int64 \$overallBudgetSecs\) 6"):
            self.assertRegex(
                deployment, pattern,
                "the chart's minimum-grace formula changed; re-derive it from "
                "grpc::transport::minimum_termination_grace")

    def test_the_paced_ending_defaults_equal_the_binarys(self):
        config = CONFIG_RS.read_text()
        values = yaml.safe_load((SERVICE_CHART / "values.yaml").read_text())
        self.assertEqual(values["shutdown"]["streamEndRate"],
                         binary_default(config, "LIVE_EDIT_SHUTDOWN_STREAM_END_RATE"))
        self.assertEqual(values["shutdown"]["streamEndMaxWindowSecs"],
                         binary_default(config, "LIVE_EDIT_SHUTDOWN_STREAM_END_MAX_WINDOW_SECS"))

    def test_default_render_passes_with_the_shutdown_settings_strategy_and_grace(self):
        docs = render()
        data = config_data(docs)
        self.assertEqual(data["LIVE_EDIT_SHUTDOWN_STREAM_END_RATE"], "25")
        self.assertEqual(data["LIVE_EDIT_SHUTDOWN_STREAM_END_MAX_WINDOW_SECS"], "20")
        spec = server_deployment(docs)["spec"]
        self.assertEqual(spec["strategy"], {
            "type": "RollingUpdate",
            "rollingUpdate": {"maxSurge": "33%", "maxUnavailable": 0}})
        # timeoutSecs + streamEndMaxWindowSecs + 5.
        self.assertEqual(spec["minReadySeconds"], 30)
        grace = spec["template"]["spec"]["terminationGracePeriodSeconds"]
        self.assertEqual(grace, 80)
        self.assertGreaterEqual(grace, binary_minimum_grace(5, 30000, 25, 20, 30))

    def test_zero_pacing_values_render_rather_than_drop(self):
        data = config_data(render({"shutdown": {"streamEndRate": 0,
                                                "streamEndMaxWindowSecs": 0}}))
        self.assertEqual(data["LIVE_EDIT_SHUTDOWN_STREAM_END_RATE"], "0")
        self.assertEqual(data["LIVE_EDIT_SHUTDOWN_STREAM_END_MAX_WINDOW_SECS"], "0")
        stderr = render({"shutdown": {"streamEndRate": None}},
                        error="shutdown.streamEndRate")
        self.assertIn("shutdown.streamEndRate must be set", stderr)

    def test_the_default_rollout_takes_at_most_three_steps(self):
        # With maxUnavailable 0 (asserted in the default-render test), each
        # step adds k surge pods, waits for them to be available, then drains
        # as many old pods: ceil(replicas / k) steps.
        max_surge = server_deployment(render())["spec"]["strategy"]["rollingUpdate"]["maxSurge"]
        for replicas in range(1, 100):
            k = surge_pods(max_surge, replicas)
            self.assertLessEqual(-(-replicas // k), 3, f"{replicas} replicas: k={k}")
            if replicas <= 3:
                self.assertEqual(k, 1, f"{replicas} replicas drain one pod at a time")

    def test_the_default_surge_headroom_matches_the_documented_table(self):
        # The values of the extra-pods table in docs/service/docs/operations.md
        # ("What to expect during a rollout") and values.yaml, copied here: a
        # change to maxSurge must update those tables with this one.
        max_surge = server_deployment(render())["spec"]["strategy"]["rollingUpdate"]["maxSurge"]
        headroom = {n: surge_pods(max_surge, n)
                    for n in (3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 20, 21)}
        self.assertEqual(headroom, {3: 1, 4: 2, 6: 2, 7: 3, 9: 3, 10: 4, 12: 4,
                                    13: 5, 15: 5, 16: 6, 18: 6, 19: 7, 20: 7, 21: 7})

    def test_the_documented_connection_requirement_follows_the_chart_defaults(self):
        # docs/service/docs/backing-store.md "Connections": the store
        # must hold 2 x autoscaling.maxReplicas replicas at the worst-case
        # per-replica count, the bounded rollout peak included, plus one
        # disaster-recovery Job.
        #
        # At rest a replica holds P0, the pooled clients opened at startup
        # (data + txn pools, the snapshot lane P_a + P_f + 4, maintenance and
        # pub/sub), one connection per primary each, plus one connection per
        # stream reader, to its bound slot's primary alone. No reader holds a
        # backchannel (the readers run fred's default Blocking::Block, b.9sh).
        #
        # At the peak, on a cluster, every pooled client can be running a
        # topology sync at once (a cluster-wide resync), and each sync holds
        # one more connection, for its CLUSTER SLOTS, while it runs (the
        # vendored fred drops it when the sync ends, change 11). connect_any
        # opens it to the first node of a sorted set, so all of a replica's
        # land on one node: at most P0 more per replica on that node, and P0
        # more cluster-wide. A standalone store runs no sync.
        #
        # On top of the two fleets, one disaster-recovery restore or
        # accept-loss Job (`run_dr_job` in src/main.rs → ValkeyClient::connect).
        # It reads the server's settings ConfigMap (test-dr-chart.py pins the
        # Job's envFrom and env to the server's), so it opens the same P0
        # pooled clients; it opens no stream reader, since readers connect
        # lazily, on a first subscription (storage::stream_mux), and the Job
        # subscribes to nothing. Its own syncs add their P0 on one node, as a
        # replica's do. The two Jobs that can overlap while an attempt
        # replaces its predecessor are a transient the requirement leaves out.
        values = yaml.safe_load((SERVICE_CHART / "values.yaml").read_text())
        valkey = values["valkey"]
        self.assertIsNone(valkey["snapshotPoolSize"])
        snapshot = (valkey["catchupSnapshotReadConcurrency"]
                    + valkey["catchupFencedReadConcurrency"] + 4)
        startup = 2 * valkey["poolSize"] + snapshot + valkey["maintenancePoolSize"] + 1
        readers = valkey["subscriberPoolSize"]
        # The pool sizes the code plans for at the chart's defaults:
        # PlanInputs::reference in src/config/memory_plan.rs, which a Rust
        # test pins to this chart (plan_inputs_reference_is_the_charts_configuration).
        memory_plan = (ROOT / "src/config/memory_plan.rs").read_text()
        reference = re.search(
            r"pub const fn reference\(\) -> Self \{\s*Self \{\s*pools: PoolSizes \{\s*"
            r"data: (\d+),\s*txn: (\d+),\s*maintenance: (\d+),\s*snapshot: (\d+),\s*"
            r"readers: (\d+),", memory_plan)
        self.assertIsNotNone(reference, "PlanInputs::reference's pools moved; update this test")
        data, txn, maintenance, snapshot_pool, plan_readers = map(int, reference.groups())
        self.assertEqual(startup, data + txn + snapshot_pool + maintenance + 1,
                         "P0 from the chart is the code's general clients, the pub/sub one included")
        self.assertEqual(readers, plan_readers)
        shards = 3

        def one_replica(p, r):
            # One replica's peak with p pooled clients and r readers: one
            # node, per cluster node (its syncs' P0 on top), cluster-wide.
            return (p + r, 2 * p + r, p * (shards + 1) + r)

        at_rest = (startup + readers,           # one node
                   startup + readers,           # per cluster node
                   startup * shards + readers)  # cluster-wide
        peak = one_replica(startup, readers)
        self.assertEqual((startup,) + at_rest, (136, 392, 392, 664))
        self.assertEqual(peak, (392, 528, 800))
        max_replicas = values["autoscaling"]["maxReplicas"]
        self.assertEqual(max_replicas, 10)
        fleets = 2 * max_replicas
        self.assertEqual(tuple(fleets * n for n in peak), (7840, 10560, 16000))
        # The one DR Job, at its peak: P0 on a standalone store; on a cluster,
        # its P0 at rest on every node plus its syncs' P0 on one of them.
        job_peak = one_replica(startup, 0)
        self.assertEqual(job_peak, (136, 272, 544))
        required = tuple(fleets * n + job for n, job in zip(peak, job_peak))
        self.assertEqual(required, (7976, 10832, 16544))

        # What the backing-store page's "Connections" section publishes, read
        # from the page, so an edit that drifts from these figures fails here.
        page = (ROOT / "docs/service/docs/backing-store.md").read_text()
        connections = page.split("(live-edit-backing-store-connections)=", 1)[1]
        connections = connections.split("\n(live-edit-", 1)[0]

        def published(pattern, what):
            found = re.search(pattern, connections)
            self.assertIsNotNone(found, f"the Connections section no longer states {what}")
            return int(found.group(1).replace(",", ""))

        self_hosted_maxclients = published(
            r"\| Self-hosted Valkey, standalone or cluster \| set `maxclients (\d+)`",
            "the self-hosted maxclients row")
        amr_small_max_replicas = published(
            r"\| Azure Managed Redis `Balanced_B0` to `Balanced_B5` \| keep "
            r"`autoscaling\.maxReplicas` at (\d+) or fewer \|",
            "the AMR Balanced_B0-B5 maxReplicas row")
        self.assertRegex(
            connections,
            r"\| Azure Managed Redis `Balanced_B10` and larger, MemoryDB, ElastiCache for "
            r"Valkey \| nothing \|",
            "the Connections section no longer states the stores that need nothing")
        amr_small = published(r"([\d,]+) on `Balanced_B0` to `Balanced_B5`",
                              "the Balanced_B0-B5 connection limit")
        amr_b10 = published(r"([\d,]+) on `Balanced_B10`", "the Balanced_B10 connection limit")
        amr_b20 = published(r"([\d,]+) on `Balanced_B20`", "the Balanced_B20 connection limit")
        managed_per_node = published(r"([\d,]+) on MemoryDB and ElastiCache",
                                     "the MemoryDB / ElastiCache per-node limit")
        self.assertEqual(
            (self_hosted_maxclients, amr_small_max_replicas, amr_small, amr_b10, amr_b20,
             managed_per_node),
            (12000, 9, 15000, 30000, 75000, 65000))

        # The page's per-replica and per-Job figures at the chart defaults:
        # the peaks one_replica() above evaluates (three primaries), and the
        # per-further-primary step, P0.
        self.assertIn(
            f"one replica opens up to {peak[0]} connections on a single-primary store; on a "
            f"cluster, up to {peak[1]} to one node and {peak[2]} in all with three primaries, "
            f"plus {startup} per further primary.", connections)
        self.assertIn(
            f"A disaster-recovery Job opens up to {job_peak[0]} on a single-primary store; on a "
            f"cluster, {job_peak[1]} to one node and {job_peak[2]} in all with three primaries, "
            f"plus {startup} per further primary.", connections)
        self.assertIn(
            "Keep twice `autoscaling.maxReplicas` replicas plus one Job within each node's limit",
            connections)

        # Self-hosted: the published maxclients holds the peak, the DR Job
        # included, on one node (7,976) and per cluster node (10,832; a
        # cluster also keeps 2 cluster-bus links per other node, well inside
        # the margin). Valkey's default 10,000 does not hold a cluster node's
        # peak, which is why the page sets it.
        self.assertLessEqual(required[0], self_hosted_maxclients)
        self.assertLessEqual(required[1], self_hosted_maxclients)
        self.assertGreater(required[1], 10000)
        # AMR's 15,000-connection SKUs: the published maxReplicas fits with the
        # Job (2 x 9 x 800 + 544 = 14,944) and one more does not
        # (2 x 10 x 800 + 544 = 16,544).
        amr_small_required = 2 * amr_small_max_replicas * peak[2] + job_peak[2]
        self.assertEqual(amr_small_required, 14944)
        self.assertLessEqual(amr_small_required, amr_small)
        self.assertGreater(2 * (amr_small_max_replicas + 1) * peak[2] + job_peak[2], amr_small)
        # "Nothing" to set on the rest: Balanced_B10 (and so B20) holds the
        # default fleets' cluster-wide peak with the Job, and MemoryDB and
        # ElastiCache's per-node limit their per-node peak with it.
        self.assertLessEqual(required[2], amr_b10)
        self.assertLessEqual(required[2], amr_b20)
        self.assertLessEqual(required[1], managed_per_node)

        # The Azure page's SKU table and its maxReplicas step publish the same
        # cap: the published figure on Balanced_B0 to B5, the chart default on
        # Balanced_B10 and B20.
        azure = (ROOT / "docs/service/docs/backing-store-azure.md").read_text()
        caps = dict(re.findall(r"(?m)^\| `(Balanced_B\d+)` \|[^|]*\|[^|]*\| (\d+) \|$", azure))
        self.assertEqual(
            {sku: int(cap) for sku, cap in caps.items()},
            {"Balanced_B0": amr_small_max_replicas, "Balanced_B1": amr_small_max_replicas,
             "Balanced_B3": amr_small_max_replicas, "Balanced_B5": amr_small_max_replicas,
             "Balanced_B10": max_replicas, "Balanced_B20": max_replicas},
            "the Azure page's SKU table caps autoscaling.maxReplicas as the Connections section "
            "does")
        self.assertIn(
            f"Set `autoscaling.maxReplicas` to {amr_small_max_replicas} on `Balanced_B0` to "
            f"`Balanced_B5`; larger sizes take the chart default of {max_replicas}.",
            azure)

        # Every self-hosted sample configuration on the self-hosted page sets
        # the published maxclients, right after its noeviction policy.
        guide = (ROOT / "docs/service/docs/backing-store-self-hosted.md").read_text()
        samples = re.findall(r"(?m)^(\s*)maxmemory-policy noeviction\n(.*)$", guide)
        self.assertGreaterEqual(len(samples), 3)
        for indent, following in samples:
            self.assertEqual(following, f"{indent}maxclients {self_hosted_maxclients}",
                             "a self-hosted sample configuration must set the published "
                             "maxclients")
        self.assertEqual(
            sorted(set(re.findall(r"(?m)^\s*maxclients\s+(\d+)\s*$", guide))),
            [str(self_hosted_maxclients)],
            "every maxclients a sample sets is the published one")

    def test_min_ready_seconds_follows_the_shutdown_settings_or_its_override(self):
        spec = server_deployment(render(shutdown_values(window=10)))["spec"]
        self.assertEqual(spec["minReadySeconds"], 5 + 10 + 5)
        spec = server_deployment(render({"minReadySeconds": 12}))["spec"]
        self.assertEqual(spec["minReadySeconds"], 12)
        spec = server_deployment(render({"strategy": {"rollingUpdate": {
            "maxSurge": "25%", "maxUnavailable": "25%"}}}))["spec"]
        self.assertEqual(spec["strategy"]["rollingUpdate"],
                         {"maxSurge": "25%", "maxUnavailable": "25%"})

    def test_a_grace_below_the_minimum_fails_naming_the_values(self):
        cases = {
            # name: (shutdown_values kwargs)
            "defaults: the RPC deadline is the later term": {},
            "a window above the RPC deadline": {"window": 40},
            "pacing off: rate 0 drops the window": {"rate": 0, "window": 40},
            "pacing off: window 0": {"window": 0},
            "a deadline rounded up to whole seconds": {"deadline_ms": 45500},
            "a longer drain and budget": {"timeout": 10, "overall": 40},
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                timeout = kwargs.get("timeout", 5)
                deadline_ms = kwargs.get("deadline_ms", 30000)
                rate = kwargs.get("rate", 25)
                window = kwargs.get("window", 20)
                overall = kwargs.get("overall", 30)
                minimum = chart_minimum_grace(timeout, deadline_ms, rate, window, overall)
                binary = binary_minimum_grace(timeout, deadline_ms, rate, window, overall)
                # The chart never accepts a grace the binary's worst case
                # exceeds, and rounds up by at most a second.
                self.assertIn(minimum - binary, (0, 1), (minimum, binary))
                # At the minimum the render passes ...
                spec = server_deployment(render(shutdown_values(minimum, **kwargs)))["spec"]
                self.assertEqual(spec["template"]["spec"]["terminationGracePeriodSeconds"],
                                 minimum)
                # ... and one second under it fails, naming every value.
                stderr = render(shutdown_values(minimum - 1, **kwargs),
                                error="terminationGracePeriodSeconds")
                for needle in (
                        f"terminationGracePeriodSeconds ({minimum - 1})",
                        f"below the {minimum} s",
                        f"shutdown.timeoutSecs ({timeout})",
                        f"ceil(rpc.deadlineMs / 1000) ({math.ceil(deadline_ms / 1000)})",
                        "shutdown.streamEndMaxWindowSecs "
                        f"({0 if rate == 0 else window}; 0 when shutdown.streamEndRate is 0)",
                        f"shutdown.overallBudgetSecs ({overall})",
                        f"at least {minimum}"):
                    self.assertIn(needle, stderr)

    def test_the_grace_check_computes_with_the_digits_the_binary_receives(self):
        # A quoted " 60" budget is 60 to the binary, so the minimum is
        # 5 + max(30, 20) + 6 + 60 + 6 = 107 and a grace of 80 is too short.
        # The old check read the string as 0 and passed it.
        stderr = render({"shutdown": {"overallBudgetSecs": " 60"},
                         "terminationGracePeriodSeconds": 80},
                        error="terminationGracePeriodSeconds")
        for needle in ("terminationGracePeriodSeconds (80)", "below the 107 s",
                       "shutdown.overallBudgetSecs (60)", "at least 107"):
            self.assertIn(needle, stderr)
        self.assertEqual(chart_minimum_grace(5, 30000, 25, 20, 60), 107)
        # A quoted " 80" grace is 80, above the default minimum of 77; the old
        # check read it as 0 and refused it.
        spec = server_deployment(render({"terminationGracePeriodSeconds": " 80"}))["spec"]
        self.assertEqual(spec["template"]["spec"]["terminationGracePeriodSeconds"], 80)

    def test_a_min_ready_seconds_that_is_not_a_whole_number_fails_naming_it(self):
        # The old chart read "30s" as 0, so a pod counted as available at
        # once and the rollout no longer waited out the previous drain.
        stderr = render({"minReadySeconds": "30s"}, error="minReadySeconds")
        self.assertIn("minReadySeconds must be a non-negative whole number (", stderr)

    def test_a_quoted_nonzero_rate_keeps_the_window_in_the_minimum(self):
        # " 25" is 25 to the binary, so pacing is on and the 40 s window is
        # the later term: 5 + max(30, 40) + 6 + 30 + 6 = 87. The old check
        # read the string as 0, dropped the window and accepted 86.
        values = shutdown_values(86, window=40)
        values["shutdown"]["streamEndRate"] = " 25"
        stderr = render(values, error="terminationGracePeriodSeconds")
        self.assertIn("below the 87 s", stderr)
        self.assertEqual(chart_minimum_grace(5, 30000, 25, 40, 30), 87)
        values["terminationGracePeriodSeconds"] = 87
        spec = server_deployment(render(values))["spec"]
        self.assertEqual(spec["template"]["spec"]["terminationGracePeriodSeconds"], 87)

    def test_the_derived_min_ready_seconds_reads_the_digits_the_binary_receives(self):
        # timeoutSecs " 10" + streamEndMaxWindowSecs 20 + 5; the old
        # derivation read the string as 0 and gave 25. Grace 90 covers the
        # minimum of 10 + 30 + 6 + 30 + 6 = 82.
        spec = server_deployment(render({"shutdown": {"timeoutSecs": " 10"},
                                         "terminationGracePeriodSeconds": 90}))["spec"]
        self.assertEqual(spec["minReadySeconds"], 35)

    def test_a_values_file_grace_of_a_million_renders_as_digits(self):
        # Helm decodes a values-file number as float64; rendered as is, it
        # reads 2e+06, which Kubernetes refuses as an integer field.
        spec = server_deployment(render({"terminationGracePeriodSeconds": 2000000}))["spec"]
        self.assertEqual(spec["template"]["spec"]["terminationGracePeriodSeconds"], 2000000)


if __name__ == "__main__":
    unittest.main()
