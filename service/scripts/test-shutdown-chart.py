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
        deployment = (SERVICE_CHART / "templates/deployment.yaml").read_text()
        self.assertRegex(
            deployment,
            r"\$minGrace := add \(int64 \$shutdown\.timeoutSecs\) "
            r"\(max \$deadlineSecs \$streamEndWindowSecs\) 6 "
            r"\(int64 \$shutdown\.overallBudgetSecs\) 6",
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
        # per-replica count: data + txn pools, the snapshot lane (P_a + P_f + 4),
        # maintenance and pub/sub at startup, plus every stream reader and its
        # backchannel. The figures and limits below are the ones the docs state.
        values = yaml.safe_load((SERVICE_CHART / "values.yaml").read_text())
        valkey = values["valkey"]
        self.assertIsNone(valkey["snapshotPoolSize"])
        snapshot = (valkey["catchupSnapshotReadConcurrency"]
                    + valkey["catchupFencedReadConcurrency"] + 4)
        startup = 2 * valkey["poolSize"] + snapshot + valkey["maintenancePoolSize"] + 1
        readers = valkey["subscriberPoolSize"]
        fleets = 2 * values["autoscaling"]["maxReplicas"]
        single_node = startup + 2 * readers
        per_cluster_node = 2 * (startup + readers)
        cluster_wide = startup * (3 + 1) + 2 * readers  # 3 shards
        self.assertEqual((startup, single_node, per_cluster_node, cluster_wide),
                         (136, 648, 784, 1056))
        self.assertEqual(fleets * single_node, 12960)
        self.assertEqual(fleets * per_cluster_node, 15680)
        self.assertEqual(fleets * cluster_wide, 21120)
        # Self-hosted maxclients 20000 covers both self-hosted topologies, and
        # Balanced_B10's 30,000 the cluster-wide figure.
        self.assertLessEqual(fleets * per_cluster_node, 20000)
        self.assertLessEqual(fleets * cluster_wide, 30000)
        # AMR's 15,000-connection SKUs: autoscaling.maxReplicas at most 7.
        self.assertLessEqual(2 * 7 * cluster_wide, 15000)
        self.assertGreater(2 * 8 * cluster_wide, 15000)
        # Every self-hosted sample configuration on the self-hosted page sets the
        # maxclients the backing-store page's "Connections" table requires.
        guide = (ROOT / "docs/service/docs/backing-store-self-hosted.md").read_text()
        self.assertGreaterEqual(guide.count("maxmemory-policy noeviction\n"), 3)
        self.assertEqual(guide.count("maxmemory-policy noeviction\n"),
                         guide.count("maxmemory-policy noeviction\n  maxclients 20000\n")
                         + guide.count("maxmemory-policy noeviction\n    maxclients 20000\n"))

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


if __name__ == "__main__":
    unittest.main()
